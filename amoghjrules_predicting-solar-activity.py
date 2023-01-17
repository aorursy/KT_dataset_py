import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/multilabel-solar-flare-dataset/C_training.csv")

df_test = pd.read_csv("../input/multilabel-solar-flare-dataset/C_testing.csv")

# M5_testing = pd.read_csv("../input/multilabel-solar-flare-dataset/M5_testing.csv")

# M5_training = pd.read_csv("../input/multilabel-solar-flare-dataset/M5_training.csv")

# M_testing = pd.read_csv("../input/multilabel-solar-flare-dataset/M_testing.csv")

# M_training = pd.read_csv("../input/multilabel-solar-flare-dataset/M_training.csv")
print(df.shape)

print(df_test.shape)

print(df['label'].describe())

print(df_test['label'].describe())
display(df.head())

display(df_test.head())
def encode(string):

    if string=="Negative":

        return -1

    elif string=='Positive':

        return +1
df = df.rename(columns = {'label' : 'target'})

df_test = df_test.rename(columns = {'label' : 'target'})

target = df['target']

target_test = df_test['target']

df.drop(['target','flare','timestamp','NOAA','HARP'], inplace = True, axis =1)

df_test.drop(['target','flare','timestamp','NOAA','HARP'], inplace = True, axis =1)

target = target.apply(encode)

target_test = target_test.apply(encode)

train = df

test = df_test
sns.countplot(target).set_title('Target distribution')

plt.show()

sns.countplot(target_test).set_title('Target distribution')
sns.heatmap(df.corr()).set_title('Correlation heatmap between features')
sns.distplot(df['USFLUX']).set_title("Distribution of the USFLUX feature")
sns.distplot(df['TOTBSQ']).set_title("Distribution of TOTBSQ feature")
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LogisticRegression

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostClassifier 

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold, cross_validate

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
n_folds = 5

def auc_score(model):

    kf = KFold( n_folds, shuffle= True).get_n_splits(train.values)

    score = cross_val_score(model, train.values, target, scoring = "roc_auc", cv = kf)

    return score
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
glm = LogisticRegression( random_state=1, solver='lbfgs', max_iter=2020, fit_intercept=True, penalty='none', verbose=0)
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(train.values, target.values)

print(model_lgb.feature_importances_)

plt.plot(model_lgb.feature_importances_)

plt.title("Feature importances according to LGBM")

plt.show()
score = auc_score(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = auc_score(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = auc_score(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

score = auc_score(glm)

print("Logistic Regression: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))

# score = auc_score(KRR)

# print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

# score = auc_score(GBoost)

# print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = auc_score(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class average_stacking(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self,models):

        self.models = models

    def fit(self, x,y):

        self.model_clones = [clone(x) for x in self.models]

        

        for model in self.model_clones:

            model.fit(x,y)

        return self

    def predict(self, x):

        preds = np.column_stack([

            model.predict(x) for model in self.model_clones

        ])

        return np.mean(preds, axis = 1)
averaged_models = average_stacking(models = (ENet, glm, model_lgb, lasso, model_xgb))



score = auc_score(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
# Dividing the trainset using a stratified kfold split, we are generating a train set. Next, we generate a test set which consists of the predictions of the pre-existing models



def generate_oof_trainset( train, target, test, test_target, strat_kfold, models,):



    oof_train = pd.DataFrame() # Initializing empty data frame

    

    count = 0

    print(train.shape, target.shape)



    for train_id, test_id in strat_kfold.split(train, target):

        count += 1

        print("Current fold number is :", count)

        xtrain, xtest = train.iloc[train_id], train.iloc[test_id]

        ytrain, ytest = target.iloc[train_id], target.iloc[test_id]

        

        curr_split = [None]*(len(models)+1) # Initializing list of lists to save all predictions for a split from all models for the current split

        

        for i in tqdm(range(len(models))):

            

            model = models[i]

            model.fit(xtrain, ytrain)

            

            curr_split[i] = model.predict_proba(xtest)[:,1]      

            

        curr_split[-1] = ytest

        oof_train = pd.concat([oof_train,pd.DataFrame(curr_split).T], ignore_index= True)

    

    oof_test = [None]*len(models)

    for i, model in enumerate(models):

        model.fit( train, target)

        oof_test[i] = model.predict_proba(test)[:,1]

    oof_test = pd.DataFrame(oof_test)

    return oof_train, oof_test
# we fit the generated trainset and perform cross validation.

from tqdm import tqdm

strat_kfold = StratifiedKFold( n_splits = 10, shuffle = True)



log_reg = LogisticRegression(max_iter= 1000, random_state = 0)

gbr = GradientBoostingClassifier(

        max_depth=6,

        n_estimators=35,

        warm_start=False,

        random_state=42)

adar = AdaBoostClassifier(n_estimators=100, random_state=0)



models = [ log_reg, gbr, adar, glm ]

train_generated, test_generated = generate_oof_trainset( train, target, test, target_test, strat_kfold, models)
lr_clf = LogisticRegression()

target = train_generated[train_generated.columns[-1]]

train_generated.drop([train_generated.columns[-1]], axis = 1 , inplace = True)



cv_results = cross_validate(lr_clf,

                            train_generated.values,

                            target.values,

                            cv = 3,

                            scoring = 'roc_auc',

                            verbose = 1,

                            return_train_score = True,

                            return_estimator = True)



print("Fit time :", cv_results['fit_time'].sum(),"secs")

print("Score time :", cv_results['score_time'].sum(),"secs")

print("Test score :", cv_results['test_score'].mean())  
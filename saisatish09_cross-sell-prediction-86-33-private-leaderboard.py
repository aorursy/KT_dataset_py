import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

from imblearn.over_sampling import RandomOverSampler



from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
train = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/train.csv')

test = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/test.csv')

submission = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')
train.head()
train.isnull().sum()
train['Response'].value_counts()
sns.countplot(train.Response)
#converting Text Object to int type

train['Vehicle_Age']=train['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

train['Gender']=train['Gender'].replace({'Male':1,'Female':0})

train['Vehicle_Damage']=train['Vehicle_Damage'].replace({'Yes':1,'No':0})

test['Vehicle_Age']=test['Vehicle_Age'].replace({'< 1 Year':0,'1-2 Year':1,'> 2 Years':2})

test['Gender']=test['Gender'].replace({'Male':1,'Female':0})

test['Vehicle_Damage']=test['Vehicle_Damage'].replace({'Yes':1,'No':0})
train.head()
test.head()
x_train = train.drop(['id', 'Response'], axis = 1)

y_train = train['Response']

test = test.drop(['id'], axis = 1)
corrmat = x_train.corr()

fig, ax = plt.subplots()

fig.set_size_inches(13,13)

sns.heatmap(corrmat, annot = True)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from sklearn.feature_selection import SelectKBest, SelectPercentile
mi = mutual_info_classif(x_train.fillna(0), y_train)

mi
# let's add the variable names and order the features

# according to the MI for clearer visualisation

mi = pd.Series(mi)

mi.index = x_train.columns

mi.sort_values(ascending=False)
# and now let's plot the ordered MI values per feature

mi.sort_values(ascending=False).plot.bar(figsize=(15, 8))
x_train['Vintage_m'] = x_train['Vintage']/30.25

test['Vintage_m'] = test['Vintage']/30.25
x_train = x_train.apply(np.ceil)

test = test.apply(np.ceil)
x_train.head()
mi = mutual_info_classif(x_train.fillna(0), y_train)

mi
# let's add the variable names and order the features

# according to the MI for clearer visualisation

mi = pd.Series(mi)

mi.index = x_train.columns

mi.sort_values(ascending=False)
# and now let's plot the ordered MI values per feature

mi.sort_values(ascending=False).plot.bar(figsize=(15, 8))
x_train_2 = x_train.drop(['Vintage'], axis = True)

test_2 = test.drop(['Vintage'], axis = True)
x_train_2.head()
x_train_2 = x_train_2.astype(int)

test_2 = test_2.astype(int)
x_train_2.head()
test_2.head()
# categorical column 

cat_col=['Gender', 'Vintage_m', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage','Policy_Sales_Channel']
x_train_2['type'] = 'train'

test_2['type'] = 'test'
all_data = pd.concat([x_train_2, test_2])
all_data.head()
# Numerical Column

numerical_cols = ['Annual_Premium', 'Age']



scaler = MinMaxScaler()

all_data[numerical_cols] = scaler.fit_transform(all_data[numerical_cols])



all_data.head()

df_train = all_data[all_data['type'] == 'train']

df_test = all_data[all_data['type'] == 'test']
df_train = df_train.drop(['type'], axis = 1)

df_test = df_test.drop(['type'], axis = 1)
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(df_train, y_train, test_size=.25, random_state=42,stratify=y_train,shuffle=True)
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

catb = CatBoostClassifier(eval_metric = 'AUC')

catb= catb.fit(X_train, Y_train,cat_features=cat_col,eval_set=(X_test, Y_test),early_stopping_rounds=50,verbose=1000)

y_cat = catb.predict(X_test)

probs_cat_train = catb.predict_proba(X_train)[:, 1]

probs_cat_test = catb.predict_proba(X_test)[:, 1]

roc_auc_score(Y_train, probs_cat_train)

roc_auc_score(Y_test, probs_cat_test)
col=['Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured', 'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage_m']
submmission = pd.read_csv('/kaggle/input/health-insurance-cross-sell-prediction/sample_submission.csv')
cat_pred= catb.predict_proba(df_test[col])[:, 1]

submmission['Response']=cat_pred

submmission.to_csv("cat_after_FE.csv", index = False)
submmission.head()
import warnings

warnings.filterwarnings("ignore")

import lightgbm as lgb

from bayes_opt import BayesianOptimization



def bayes_parameter_opt_lgb(X_train, Y_train, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):

    train_data = lgb.Dataset(data=X_train, label=Y_train, categorical_feature = cat_col, free_raw_data=False)

    # parameters

    def lgb_eval(num_leaves,min_child_samples, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):

        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}

        params["num_leaves"] = int(round(num_leaves))

        params["min_child_samples"] = int(round(min_child_samples))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['lambda_l1'] = max(lambda_l1, 0)

        params['lambda_l2'] = max(lambda_l2, 0)

        params['min_split_gain'] = min_split_gain

        params['min_child_weight'] = min_child_weight

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, shuffle=True, verbose_eval =200, metrics=['auc'])

        return max(cv_result['auc-mean'])

    # range 

    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),

                                        'min_child_samples': (10,500),

                                        'feature_fraction': (0.1, 0.9),

                                        'bagging_fraction': (0.5, 1),

                                        'max_depth': (5, 9),

                                        'lambda_l1': (0, 5),

                                        'lambda_l2': (0, 3),

                                        'min_split_gain': (0.001, 0.1),                                        

                                        'min_child_weight': (5, 50)}, random_state=0)

# optimize

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)



    model_auc=[]

    for model in range(len( lgbBO.res)):

        model_auc.append(lgbBO.res[model]['target'])



    # return best parameters

    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']

opt_params = bayes_parameter_opt_lgb(X_train, Y_train, init_round=20, opt_round=50, n_folds=3, random_seed=6, n_estimators=1000, learning_rate=0.05)
(opt_params)
opt_parameters = opt_params[1]
opt_parameters["bagging_fraction"]
import lightgbm as lgb

opt_model_lgb = lgb.LGBMClassifier(bagging_fraction = opt_parameters["bagging_fraction"],

                                   feature_fraction = opt_parameters["feature_fraction"],

                                   lambda_l1 = opt_parameters["lambda_l1"],

                                   lambda_l2 = opt_parameters["lambda_l2"],

                                   max_depth = int(round(opt_parameters["max_depth"])),

                                   min_child_weight = opt_parameters["min_child_weight"],

                                   min_child_samples = int(round(opt_parameters["min_child_samples"])),

                                   min_split_gain = opt_parameters["min_split_gain"],

                                   num_leaves = int(round(opt_parameters["num_leaves"]))

                                    )
opt_model_lgb.fit(X_train,Y_train)
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.metrics import roc_auc_score

probs_lgb_train = opt_model_lgb.predict_proba(X_train)[:, 1]

probs_lgb_test = opt_model_lgb.predict_proba(X_test)[:, 1]

roc_auc_score(Y_train, probs_lgb_train)

roc_auc_score(Y_test, probs_lgb_test)

pred_lgb = opt_model_lgb.predict_proba(X_test)

roc_auc_score(Y_test, probs_lgb_test)
lgb_pred= opt_model_lgb.predict_proba(df_test[col])[:, 1]

submmission['Response']=lgb_pred

submmission.to_csv("lgb_after_FE.csv", index = False)
average_pred = (cat_pred + lgb_pred)/2
average_pred[:5]
submmission['Response']=average_pred

submmission.to_csv("Stacking_of_CAT_LGB_after_FE.csv", index = False)
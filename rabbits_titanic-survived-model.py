import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
# /kaggle/input/titanic/train.csv
# /kaggle/input/titanic/gender_submission.csv
# /kaggle/input/titanic/test.csv

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train.head()
corrmat = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, annot=True, fmt='.2f', vmax=1, square=True)
sns.set()
sns.pairplot(train, size = 2.5)
plt.show()
sns.barplot(train['Survived'], train['Sex'])
sns.swarmplot(train['Survived'], train['Fare'])
sns.barplot(train['Survived'], train['Pclass'])
sns.swarmplot(x="Survived", y="Fare", data=train)
sns.swarmplot(train['Survived'], train['Age'])
# sns.barplot(train['Survived'], train['Age'])
sns.barplot(train['Survived'], (train['Age'] < 15).astype(int))

train_survived = train['Survived']
train_len = len(train)
all_data = pd.concat((train, test), ignore_index=True)

train_len
# FamilyName, Title, Famliy
all_data['FamilyName'] = all_data['Name'].apply(lambda st: st[0:st.find(",")])
all_data['Title'] = all_data['Name'].apply(lambda st: st[st.find(",") + 1:st.find(".")])
all_data['Family'] = all_data['SibSp'] + all_data['Parch'] > 0
all_data['Family'] = all_data['Family'].astype(int)
# IsChild
# all_data['IsChild'] = (all_data['Age'] < 20).astype(int)

# Sex (trans to int)
# all_data['Sex'] = (all_data['Sex'] == 'female').astype(int)

# all_data.tail()
# missing_train
missing = all_data.isnull().sum().sort_values(ascending=False)
missing
# Cabin => None
all_data['Cabin'] = all_data['Cabin'].fillna('None')

# Age => Drop
all_data.drop(['Age'], axis=1, inplace=True)
# Fare => mean (Pclass, Embarked, Famliy)
fare_nan_data = all_data[all_data['Fare'].isnull()]
fare_nan_data_id = fare_nan_data['PassengerId']
fare_nan_data

fare_nan_data_input = all_data[(all_data['Embarked'] == fare_nan_data['Embarked'].values[0]) & 
                               (all_data['Family'] == fare_nan_data['Family'].values[0]) & 
                            (all_data['Pclass'] == fare_nan_data['Pclass'].values[0])].mean()
all_data.loc[fare_nan_data.index, 'Fare'] = fare_nan_data_input['Fare']
all_data.loc[fare_nan_data.index, 'Fare']
# Embarked => Drop        
all_data[all_data['Embarked'].isnull()]
all_data = all_data.drop(index=[61, 829])
all_data.head()
# Scaler(Age, Fare)
scaler = MinMaxScaler()
all_data['Fare'] = scaler.fit_transform(all_data['Fare'].to_numpy().reshape(-1, 1))
all_data.head()
all_data['Sex'] = all_data['Sex'] == 'female'
all_data['Sex'] = all_data['Sex'].astype(int)
all_data.head()



all_data.drop(['Name', 'SibSp', 'Parch', 'Cabin', 'FamilyName', 'Embarked', 'Title'], axis=1, inplace=True)
all_data.head()
# all_data['Pclass'] = all_data['Pclass'].astype(str)

all_data = pd.get_dummies(all_data)
all_data.head()
# Split Train and Test
train = all_data[:train_len - 2] # remove 2 rows (by Embarked)
test = all_data[train_len - 2:]

# Test Id
test_id = test['PassengerId']
y_train = train.Survived.values

# Drop Id
train.drop(['PassengerId'], axis=1, inplace=True)
test.drop(['PassengerId'], axis=1, inplace=True)

# Drop Survived
train.drop(['Survived'], axis=1, inplace=True)
test.drop(['Survived'], axis=1, inplace=True)

print(len(all_data), len(train), len(test))
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
ridge = make_pipeline(RobustScaler(), Ridge(alpha =0.0005, random_state=1))
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost1 = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
# GBoost with different option
GBoost2 = GradientBoostingRegressor(n_estimators=6000,
                                learning_rate=0.01,
                                max_depth=4,
                                max_features='sqrt',
                                min_samples_leaf=15,
                                min_samples_split=10,
                                loss='huber',
                                random_state=5)
RF = make_pipeline(RobustScaler(), RandomForestRegressor(n_estimators=1200,
                          max_depth=15,
                          min_samples_split=5,
                          min_samples_leaf=5,
                          max_features=None,
                          oob_score=True,
                          random_state=5))
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
ridge_score = rmsle_cv(ridge)
lasso_score = rmsle_cv(lasso)
enet_score = rmsle_cv(ENet)
krr_score = rmsle_cv(KRR)
gboost1_score = rmsle_cv(GBoost1)
gboost2_score = rmsle_cv(GBoost2)
rf_score = rmsle_cv(RF)
xgb_score = rmsle_cv(model_xgb)
lgb_score = rmsle_cv(model_lgb)
print("Ridge score: {:.4f} ({:.4f})\n".format(ridge_score.mean(), ridge_score.std()))
print("Lasso score: {:.4f} ({:.4f})\n".format(lasso_score.mean(), lasso_score.std()))
print("ENet score: {:.4f} ({:.4f})\n".format(enet_score.mean(), enet_score.std()))
print("KRR score: {:.4f} ({:.4f})\n".format(krr_score.mean(), krr_score.std()))
print("Gradient Boosting1 score: {:.4f} ({:.4f})\n".format(gboost1_score.mean(), gboost1_score.std()))
print("Gradient Boosting2 score: {:.4f} ({:.4f})\n".format(gboost2_score.mean(), gboost2_score.std()))
print("Random Forest score: {:.4f} ({:.4f})\n".format(rf_score.mean(), rf_score.std()))
print("Xgboost score: {:.4f} ({:.4f})\n".format(xgb_score.mean(), xgb_score.std()))
print("LGBM score: {:.4f} ({:.4f})\n" .format(lgb_score.mean(), lgb_score.std()))
print("totalAVG: {:.4f}\n".format((ridge_score.mean() +
                                   lasso_score.mean() +
                                   enet_score.mean()+
                                   krr_score.mean() +
                                   gboost1_score.mean() +
                                   gboost2_score.mean() +
                                   rf_score.mean() +
                                   xgb_score.mean() +
                                   lgb_score.mean()) / 9))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=200)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, KRR, GBoost1, lasso),
                                                 meta_model = RF)

score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = stacked_averaged_models.predict(test.values)
print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = model_xgb.predict(test)
print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = model_lgb.predict(test.values)
print(rmsle(y_train, lgb_train_pred))
ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
# print(ensemble)
for n, i in enumerate(ensemble):
    if i >= 0.5:
        ensemble[n] = 1
    else:
        ensemble[n] = 0
print(ensemble)
result = pd.DataFrame()
result['PassengerId'] = test_id
result['Survived'] = np.asarray(ensemble, dtype=int)

result.head()
result.to_csv('submission_sex_to_num.csv', index=False)
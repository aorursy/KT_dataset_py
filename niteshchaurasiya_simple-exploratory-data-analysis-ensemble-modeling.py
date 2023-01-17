%matplotlib inline
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor

from catboost import CatBoostRegressor

from sklearn.linear_model import Lasso,Ridge

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler,MinMaxScaler

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

print(train.shape,test.shape)
pd.pandas.set_option('display.max_columns',None)

train.head()
na_columns = [col for col in train.columns if train[col].isna().sum()>=1]

# percentage of missing values

for col in na_columns:

    print(col,np.round(train[col].isna().mean(),4))
for col in na_columns:

    df = train.copy()

    df[col] = np.where(df[col].isna(),1,0)

    #median salesprice for missing values and non missing values

    fig = px.bar(y = df.groupby(col)['SalePrice'].mean(),width=500,height = 500,template="plotly_dark",labels={'x':col,'y':'Mean Saleprice'})

    fig.show()
numerical_col = [col for col in train.columns if train[col].dtypes !='O']

print(numerical_col)

print(len(numerical_col))
df[numerical_col].head()
year_col = ['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold']

fig = px.line(train.groupby('YrSold')['SalePrice'].median(),width=600,height=400,template='plotly_dark')

fig.show()
for col in year_col:

    if col != 'YrSold':

        df = train.copy()

        df[col] = df['YrSold']-df[col]

        fig = px.scatter(x = df[col],y = df['SalePrice'],width=650,height=450,labels={'x':col,'y':'Saleprice'},template="plotly_dark")

        fig.show()
descrete_col = [col for col in numerical_col if len(train[col].unique())<25 and col not in year_col]

len(descrete_col)
train[descrete_col].head()
for col in descrete_col:

    df = train.copy()

    plt.style.use('ggplot')

    with plt.style.context('dark_background'):

        df.groupby(col)['SalePrice'].median().plot.bar(color = 'c')

    plt.ylabel('Median SalePrice')

    plt.show()
conti_col = [col for col in numerical_col if col not in descrete_col+year_col+['Id']]

print(len(conti_col),'continuous features')

print(conti_col)
for col in conti_col:

    df = train.copy()

    fig = px.histogram(x = col,data_frame= df,width=600,height=450,template='plotly_dark')

    fig.show()
for col in conti_col:

    df = train.copy()

    if 0 in df[col].unique():

        pass

    else:

        df[col] = np.log(df[col])

        fig = px.scatter(x = df[col],y = np.log(df['SalePrice']),labels = {'x' : col,'y':'Saleprice'},width=650,height=400,template='plotly_dark')

        fig.show()
for col in conti_col:

    df = train.copy()

    if 0 in df[col].unique():

        pass

    else:

        df[col] = np.log(df[col])

        fig = px.box(data_frame=df,y = df[col],labels={'x':col,'y': col},width=500,height=400,template = 'plotly_dark')

        fig.show()
cat_col = [col for col in train.columns if train[col].dtypes == 'O']

print(cat_col)

print('We have {} categorical features'.format(len(cat_col)))
train[cat_col].head()
for col in cat_col:

    print(col,': {}'.format(train[col].nunique()))
for col in cat_col:

    df = train.copy()

    fig = px.bar(df.groupby(col)['SalePrice'].mean(),height=400,width=600,template='plotly_dark')

    fig.show()
train.head()
test.tail()
y = train['SalePrice'].reset_index(drop=True)

previous_train = train.copy()
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
##Let's first handle categorical variables which are missing

nan_feature = [col for col in all_data.columns if all_data[col].isnull().sum()>=1 and all_data[col].dtypes == 'O']

print(len(nan_feature))

for feture in nan_feature:

    print('{} has {} % missing values'.format(feture,np.round(all_data[feture].isnull().mean(),4)))
#replace missing values with a new label

def replace_missing(data,missing_feature):

    df = data.copy()

    df[missing_feature] = df[missing_feature].fillna('missing')

    return df



all_data = replace_missing(all_data,nan_feature)

all_data[nan_feature].isnull().sum()
## Now let's check the numerical columns contaning missing values

num_na_col = [col for col in all_data.columns if all_data[col].isnull().sum()>=1 and all_data[col].dtypes != 'O']

for col in num_na_col:

    print('{} has {}% null values'.format(col,np.round(all_data[col].isnull().mean(),4)))
#replacing by median since data contain outliers

for col in num_na_col:

    median_value = all_data[col].median()

    

    all_data[col+'NaN'] = np.where(all_data[col].isnull(),1,0)##New Feature to capture missing value

    all_data[col].fillna(median_value,inplace = True) ##Replacing with median in corresponding feature 

    

all_data[num_na_col].isnull().sum()
for col in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

    all_data[col] = all_data['YrSold']-all_data[col]

    

all_data[year_col].head()
num_col = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea']



for col in num_col:

    all_data[col] = np.log1p(all_data[col])
## Handling Rare lables occuring in categorical variables

cat_col = [col for col in all_data.columns if all_data[col].dtypes == 'O']

for col in cat_col:

    temp = all_data[col].value_counts()/len(all_data)

    temp_df = temp[temp>0.01].index

    all_data[col] = np.where(all_data[col].isin(temp_df),all_data[col],'RareVar')
all_data.head()
all_data.shape
cat_train = all_data.iloc[:len(y), :]



cat_test = all_data.iloc[len(y):, :]
cat_train.head()
cat_test.tail()
all1 = pd.get_dummies(all_data,drop_first=True)
all1.shape
x_train = all1.iloc[:len(y), :]



x_test = all1.iloc[len(y):, :]
x_train.head()
x_test.head()
x_train.drop('Id',axis = 1,inplace = True)

x_test.drop('Id',axis = 1,inplace = True)

cat_train.drop('Id',axis = 1,inplace = True)

cat_test.drop('Id',axis = 1,inplace = True)
x_train.describe()
y_transformed = np.log1p(y)
y_transformed
kf = KFold(n_splits=9, random_state=42, shuffle=True)
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X,y_true):

    rmse = np.sqrt(-cross_val_score(model, X, y_true, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
lasso_reg =make_pipeline(Min(),Lasso(alpha=0.0005,random_state=44))
scores = {}



score = cv_rmse(lasso_reg,x_train,y_transformed)

print("Lasso: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['Lasso'] = (score.mean(), score.std())
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)
scores = {}



score = cv_rmse(GBoost,x_train,y_transformed)

print("gboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gboost'] = (score.mean(), score.std())
model_xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213,

                             random_state =7, nthread = -1)
scores = {}



score = cv_rmse(model_xgb,x_train,y_transformed)

print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['xgboost'] = (score.mean(), score.std())
model_lgb = LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
scores = {}



score = cv_rmse(model_lgb,x_train,y_transformed)

print("lgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lgbm'] = (score.mean(), score.std())
ridge = make_pipeline(RobustScaler(),Ridge(alpha = 0.0005,random_state = 12))
scores = {}



score = cv_rmse(ridge,x_train,y_transformed)

print("ridge: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['ridge'] = (score.mean(), score.std())
bag = BaggingRegressor(base_estimator=model_lgb,n_estimators=2,random_state=2,n_jobs=-1)
scores = {}



score = cv_rmse(bag,x_train,y_transformed)

print("bag: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['bag'] = (score.mean(), score.std())
print('Lightgbm')

model_lgb.fit(x_train,y_transformed)



print('model_xgb')

model_xgb.fit(x_train,y_transformed)



print('Gradient_boosting')

GBoost.fit(x_train,y_transformed)



print('bagging')



bag.fit(x_train,y_transformed)
def blend_models_predict(X):

    return (#(0.16  * elastic_model.predict(X)) + \

            #(0.16 * lasso.predict(X)) + \

            (0.2 * GBoost.predict(X)) + \

            (0.3 * model_lgb.predict(X)) + \

            (0.2 * model_xgb.predict(X)) + \

#             (0.1 * xgb_model_full_data.predict(X)) + \

            (0.3 * bag.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y_transformed, blend_models_predict(x_train)))
print('Predict submission')

submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = (np.expm1(blend_models_predict(x_test)))
submission.to_csv("submission2.csv", index=False)
q1 = submission['SalePrice'].quantile(0.0042)

q2 = submission['SalePrice'].quantile(0.99)

#Quantiles helping us get some extreme values for extremely low or high values 

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission3.csv", index=False)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.shape
df.isnull().sum()
pd.options.display.max_columns=100

df.head()
df.info()
df.count()
df.isna().mean().round(4)*100
df.head()
df=df.drop(['PoolQC','Alley','FireplaceQu','Fence','MiscFeature','Id'] ,axis=1)
df.head()
df['MSZoning'].value_counts()
#filling missing values
df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].median())
df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])
df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])
df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])
df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])
df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].median())


df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])

df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])
df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])
df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])
df.shape
df.isnull().sum()

df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])
df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mode()[0])
corrmat = df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
a=df.select_dtypes('object').columns[:-1]
a
df.columns.unique()
df.head()
df.columns.unique()
df.head()
df.head()
df.shape


df.head()
df.shape
df.info()
df.isnull().sum()
#using heat map to analaysing null
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df.dropna(inplace=True)
df.head()
df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df_test.head()
df_test=df_test.drop(['PoolQC','Alley','FireplaceQu','Fence','MiscFeature','Id'] ,axis=1)
df_test.head()
#filling misssing values
df_test['BsmtCond']=df_test['BsmtCond'].fillna(df_test['BsmtCond'].mode()[0])
df_test['BsmtQual']=df_test['BsmtQual'].fillna(df_test['BsmtQual'].mode()[0])
df_test['BsmtExposure']=df_test['BsmtExposure'].fillna(df_test['BsmtExposure'].mode()[0])
df_test['BsmtFinType1']=df_test['BsmtFinType1'].fillna(df_test['BsmtFinType1'].mode()[0])

df_test['BsmtFinType2']=df_test['BsmtFinType2'].fillna(df_test['BsmtFinType2'].mode()[0])
df_test['GarageYrBlt']=df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].median())
df_test['MasVnrType']=df_test['MasVnrType'].fillna(df_test['MasVnrType'].mode()[0])
df_test['MasVnrArea']=df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mode()[0])
df_test['LotFrontage']=df_test['LotFrontage'].fillna(df_test['LotFrontage'].median())
df_test['MSZoning']=df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
df_test['GarageType']=df_test['GarageType'].fillna(df_test['GarageType'].mode()[0])
df_test['GarageFinish']=df_test['GarageFinish'].fillna(df_test['GarageFinish'].mode()[0])
df_test['GarageQual']=df_test['GarageQual'].fillna(df_test['GarageQual'].mode()[0])
df_test['GarageCond']=df_test['GarageCond'].fillna(df_test['GarageCond'].mode()[0])
df_test.info()
#checking of null value using heatmap
sns.heatmap(df.isnull(),yticklabels=False,cbar=False)
df_test.dropna(inplace=True)
df_test.shape
final_df=pd.concat([df,df_test],axis=0)
final_df.head()
final_df.shape
#dealing with numerical features
##dealing with time and date
numerical_features = [feature for feature in final_df.columns if final_df[feature].dtypes != 'O']




final_df[numerical_features].head()
year_feature = [feature for feature in final_df.columns if 'Yr' in feature or 'Year' in feature]

year_feature
for feature in year_feature:
   print(feature, df[feature].unique())
for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    final_df[feature]=final_df['YrSold']-final_df[feature]
final_df.head()
##dealing with categorical features
categorical_features=[feature for feature in final_df.columns if final_df[feature].dtype=='O']
categorical_features
for feature in categorical_features:
    labels_ordered=final_df.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    final_df[feature]=final_df[feature].map(labels_ordered)
final_df.head()
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

for feature in num_features:
    final_df[feature]=np.log(final_df[feature])
final_df.shape
df_Train=final_df.iloc[:1454,:]
df_Test=final_df.iloc[1454:,:]
df_Train.head()
df_Test.head()
X_train=df_Train.drop(['SalePrice'],axis=1)
y_train=df_Train['SalePrice']
##pridiction and algorithm
from sklearn.model_selection import RandomizedSearchCV
import xgboost
classifier=xgboost.XGBRegressor()
import xgboost
regressor=xgboost.XGBRegressor()
booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]
# Hyper Parameter Optimization


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }
random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)
random_cv.fit(X_train,y_train)

random_cv.best_estimator_
regressor=xgboost.XGBRegressor(base_score=0.25, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.1, max_delta_step=0, max_depth=2,
             min_child_weight=1, monotone_constraints='()',
             n_estimators=900, n_jobs=0, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(df_Test.drop(['SalePrice'],axis=1))
y_pred
pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sample_submission.csv',index=False)
sub_df.head()
sub_df.head()
datasets.head(10)
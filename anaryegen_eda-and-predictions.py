import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
df.head()
df.info()
df.describe()
df.isnull().sum()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(),linewidths=5,fmt='.1f',ax=ax)

plt.show()
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, PowerTransformer

from sklearn.pipeline import Pipeline, make_pipeline



class LabelEncoderBlock(BaseEstimator, TransformerMixin):

    """

        Description: Performs label encoding on pandas dataframe

        Dependency: pandas

    """

    def __init__(self, _columns, _save_nan = True):

        """

        Input: 

            _columns(list or np.array)

            _save_nan(Boolean)

        Dependency: pandas

        """

        self.__cols = _columns

        self.size = len(_columns)

        self.save_nan = _save_nan



    def fit(self, X, y = None): 

        self.present_columns = list(X.columns.values); return self

    

    def transform(self, X):

        

        columns = self.__cols

        for i in range(0, self.size):

            if columns[i] in self.present_columns:

                X.loc[:,columns[i]] = pd.factorize(X[columns[i]])[0]

                X.loc[:,columns[i]] = X.loc[:,columns[i]].astype('category')

        

        if self.save_nan == True:

          for i in range(0, self.size):

              if columns[i] in self.present_columns:

                  X.loc[:, columns[i]].replace(-1, np.NAN, inplace = True)

        return X





class Object2Label(BaseEstimator, TransformerMixin): # Valid

    """ 

    Description:  does label encoding on columns with object datatype

    """

    def fit(self, X, y = None):



        # Selecting an object column

        self.columns = list(X.select_dtypes(include=['object']).columns.values)

        self.col_amount = len(self.columns)

        if (len(self.columns) > 0):

          self.label_encoder = LabelEncoderBlock(self.columns)

          self.label_encoder.fit(X)

        return self

    

    def transform(self, X):

        if (self.col_amount > 0):

          return self.label_encoder.transform(X)

        else:

          return X





df_temp = df.copy()



# X & Y

X = df_temp.iloc[:,1:-1]

y = df_temp.iloc[:,-1]
from sklearn.linear_model import LinearRegression

from sklearn.impute import SimpleImputer

import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler



preprocessor = make_pipeline(Object2Label(),

               SimpleImputer(),

               MinMaxScaler(),

              )



X_prep = preprocessor.fit_transform(X)
model = sm.OLS(y,X_prep).fit()
cols = pd.DataFrame({

    'X': np.arange(X_prep.shape[1]),

    'Name': X.columns

})
cols.head()
model.summary()
cols.T
vals = df[['MSSubClass', 'LotArea', 'Street', 'Condition1', 'Condition2', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle',

   'RoofMatl', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtExposure', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

   'GrLivArea', 'BsmtFullBath', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 

   'Fireplaces', 'FireplaceQu', 'GarageCars', 'GarageQual', 'WoodDeckSF', 'ScreenPorch', 'PoolQC', 'SaleType', 'SalePrice']]
vals.head()
vals.info()
list(set(df.columns) - set(vals.columns))
sns.pairplot(data=vals)
sns.scatterplot(x='OverallQual', y='SalePrice', data=vals)
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=vals)
vals[vals['TotalBsmtSF'] > 6000]

vals[vals['TotalBsmtSF'] < 3000][vals['SalePrice'] > 700000]
outlier = vals[vals['TotalBsmtSF'] > 6000].index

vals.drop(outlier, inplace=True)

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=vals)
sns.scatterplot(x='Condition2', y='SalePrice', data=vals)
vals.drop('Condition2', axis=1, inplace=True)
sns.scatterplot(data=vals, x = 'FullBath', y = 'SalePrice')
sns.scatterplot(data=vals, x= 'GarageCars', y = 'SalePrice')
vals[vals['GarageCars'] == 4]
vals[vals['GarageCars'] == 3][vals['SalePrice'] >= 600000]
sns.scatterplot(vals['GrLivArea'], vals['SalePrice'])
plt.figure(figsize=(18,8))

year_graph = sns.countplot(x='YearBuilt',data=vals)

year_graph.set_xticklabels(year_graph.get_xticklabels(), rotation=90, ha="right")

plt.tight_layout()
h0 = 'Most of the houses built in 2000 and newer are above average in SalePrice'

h1 = 'Most of the houses built in 2000 and newer are not above average in SalePrice: its not the key factor'



all_new_houses = vals[vals['YearBuilt'] >= 2000].count()[0] 

amount_of_new_expensive_houses = all_new_houses - vals[vals['YearBuilt'] >= 2000][vals['SalePrice'] < vals['SalePrice'].mean()].count()[0]

proportion_of_new_exp_houses  = amount_of_new_expensive_houses / all_new_houses 

proportion_of_new_notexp_houses = (vals[vals['YearBuilt'] >= 2000].count()[0] - amount_of_new_expensive_houses) / all_new_houses 



print('All new houses: ', all_new_houses)

print('Number of new houses with above average prices: ', amount_of_new_expensive_houses)

print('Proprotion of new houses with above average prics: ', proportion_of_new_exp_houses)

print('Proportion of new houses with below average prices: ', proportion_of_new_notexp_houses, '\n')



if proportion_of_new_exp_houses > proportion_of_new_notexp_houses: 

  print(h0, '\n') 

else: 

  print(h1, '\n')
sns.scatterplot(data=vals,x='SalePrice', y='PoolQC')
vals['PoolQC'].value_counts()
vals[vals['PoolQC'].notnull()]
df['PoolArea'].nunique()
vals.drop('PoolQC', axis=1, inplace=True)
sns.scatterplot(x='FireplaceQu', y='SalePrice', data=vals)
vals.drop('FireplaceQu', axis=1, inplace=True)
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=vals)
vals[vals['MasVnrArea'].isna()]
vals['MasVnrArea'].fillna(0, inplace=True)
cat_cols = list(vals.select_dtypes(include=['object']).columns.values)

vals[cat_cols] = vals[cat_cols].astype('category')

vals.info()
vals[cat_cols] = vals[cat_cols].apply(lambda x: x.cat.codes)

X = vals.iloc[:,1:-1]

y = vals.iloc[:,-1]
vals.head()
# Import Libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
#Split data to test and train data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# Create regressor object

reg = LinearRegression() 

 

# Fitting data

reg.fit(X_train, y_train) 
# Predict

y_pred = reg.predict(X_test) 
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))
plt.scatter(y_pred, y_test, color='g')

plt.xlabel('Predicted')

plt.ylabel('Real')


linear_reg_df = pd.DataFrame(y_test)

linear_reg_df['y_pred'] = y_pred

linear_reg_df
# Plot regression model

sns.lmplot(x='SalePrice', y='y_pred', data=linear_reg_df)
# Scatterplot

sns.scatterplot(x='SalePrice', y='y_pred', data=linear_reg_df)
# import the regressor 

from sklearn.tree import DecisionTreeRegressor  

  

# create a regressor object 

tree_reg = DecisionTreeRegressor()  

  

# fit the regressor with X and Y data 

tree_reg.fit(X_train, y_train) 
y_tree_pred = tree_reg.predict(X_test) 

  

tree_reg_df = pd.DataFrame(y_test)

tree_reg_df['y_pred'] = y_tree_pred

tree_reg_df
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(y_tree_pred, y_test)))
sns.lmplot(x='SalePrice', y='y_pred', data=tree_reg_df)
from matplotlib import pyplot as plt

plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_tree_pred)

plt.plot([0, 50], [0, 50], '--k')

plt.axis('tight')

plt.xlabel('True')

plt.ylabel('Predicted')

plt.tight_layout()
from xgboost import XGBRegressor



xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)
# make predictions

xgb_pred = xgb_reg.predict(X_test)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(xgb_pred, y_test)))
xgb_reg_df = pd.DataFrame(y_test)

xgb_reg_df['y_pred'] = xgb_pred

xgb_reg_df
sns.lmplot(x="SalePrice", y='y_pred', data=xgb_reg_df)
sns.scatterplot(x='SalePrice', y='y_pred', data=xgb_reg_df)
df.fillna(0, inplace=True)
cat_cols_df = list(df.select_dtypes(include=['object']).columns.values)

df[cat_cols_df] = df[cat_cols_df].astype('category')

df.info()

df[cat_cols_df] = df[cat_cols_df].apply(lambda x: x.cat.codes)


df.head()
# X & Y

df_X = df.iloc[:,1:-1]

df_y = df.iloc[:,-1]

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression


X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size = 0.3)



reg = LinearRegression() 



reg.fit(X_train, y_train) 
# Predict

y_pred = reg.predict(X_test) 
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(y_pred, y_test)))
plt.scatter(y_pred, y_test, color='g')

plt.xlabel('Predicted')

plt.ylabel('Real')
from xgboost import XGBRegressor



xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)
# make predictions

xgb_pred = xgb_reg.predict(X_test)



from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(xgb_pred, y_test)))
xgb_reg_df_1 = pd.DataFrame(y_test)

xgb_reg_df_1['y_pred'] = xgb_pred

xgb_reg_df_1
sns.lmplot(x="SalePrice", y='y_pred', data=xgb_reg_df_1)
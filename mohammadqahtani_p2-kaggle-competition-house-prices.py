import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



import seaborn as sns



plt.style.use('fivethirtyeight')



%matplotlib inline

%config InlineBackend.figure_format = 'retina'



# Lines below are just to ignore warnings

import warnings

warnings.filterwarnings('ignore')
# loading training dataset

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
# take a look at the head of the training data set

train.head()
# shape of training dataset

train.shape
# loadinig testing dataset

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# take a look at the head 

test.head()
# the shape of the test data set

test.shape
# merging both dataset to do data cleaning on both at once, also getting more accurate filling resualt

df = train.merge(test , how='outer')

df.head()
# checking for nulls in all the columns

df.info()
# based on the discretion of the data Nan refers to inapplicability or availability

#of that feature hence it was filled with 'None' 

df[[ 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

        'SaleCondition']] = df[[ 'Street', 'Alley', 'LotShape',

                                             'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl'

                                 , 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature'

                                 , 'SaleCondition']].fillna("None")
# checking missing values for categorical variables 

print(df.MSZoning.value_counts(),

df.Electrical.value_counts(),

df.KitchenQual.value_counts(),

df.Exterior1st.value_counts(),

df.Exterior2nd.value_counts(),

df.SaleType.value_counts())
# using fill Mode technique to most data due to having high frequency of accuring

df.MSZoning.fillna(df['MSZoning'].mode()[0] , inplace = True)

df.Electrical.fillna(df['Electrical'].mode()[0] , inplace = True)

df.KitchenQual.fillna(df['KitchenQual'].mode()[0] , inplace = True)

df.Exterior1st.fillna(df['Exterior1st'].mode()[0] , inplace = True)

df.Exterior2nd.fillna(df['Exterior2nd'].mode()[0], inplace = True)

df.SaleType.fillna(df['SaleType'].mode()[0] , inplace = True)
# Fill null with 0 for some numeric columns, these columns should be zero because they dont exist in the house

df['GarageYrBlt'].fillna(0 , inplace = True)

df['GarageArea'].fillna(0, inplace = True)

df['GarageCars'].fillna(0, inplace = True)

df['BsmtFinSF1'].fillna(0, inplace = True)

df['BsmtFinSF2'].fillna(0, inplace = True)

df['BsmtUnfSF'].fillna(0, inplace = True)

df['TotalBsmtSF'].fillna(0, inplace = True)

df['BsmtFullBath'].fillna(0, inplace = True)

df['BsmtHalfBath'].fillna(0, inplace = True)

df['MasVnrArea'].fillna(0, inplace = True)
# Applying median on lot frontage based on Nieghborhood to get more accurate fill

df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
# some numeric varibles referes to date or categories so changed type to str, so that the model doesnot treat them as numeric 

# MSSubClass=The building class

df['MSSubClass'] = df['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

df['OverallCond'] = df['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

df['YrSold'] = df['YrSold'].astype(str)

df['MoSold'] = df['MoSold'].astype(str)
# No Null/Missing values, note that the nulls on the sale price due to the testing data not having a label

df.info()
# take  a look at the head

df.head()
# Plot Scatter plots of Sale Price and 4 correlated variables according to Neighborhood

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'LotArea', 'YearBuilt','Neighborhood']

plt.figure(figsize=(14,8), dpi= 80)

sns.pairplot(df[cols], kind="scatter", hue="Neighborhood",palette="RdBu")

plt.show()

# to get the train data from the meged data set we can use iloc and get all columns, while rows equal to the shape of the train[0]

df.iloc[:train.shape[0],:].head()
#saleprice correlation matrix

corrmat = df.iloc[:train.shape[0],:].corr()

plt.figure(figsize=(17, 8))

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df.iloc[:train.shape[0],:][cols].values.T)

sns.set(font_scale=1.50)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,

                 cmap = 'RdBu', linecolor = 'white', linewidth = 1)

plt.title("Correlations between Sales Price and 15 features", fontsize =15)

plt.show()
# creating dummy variables for all categorical variables in the cleaned and merged dataset

df_d = pd.get_dummies(df , drop_first=True)

df_d.shape
#saleprice correlation matrix

corrmat = df_d.iloc[:train.shape[0],:].corr()

plt.figure(figsize=(17, 8))

k = 15 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_d.iloc[:train.shape[0],:][cols].values.T)

sns.set(font_scale=1.50)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True,

                 fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values,

                 cmap = 'RdBu', linecolor = 'white', linewidth = 1)

plt.title("Correlations between Sales Price and 15 features", fontsize =15)

plt.show()
# getting the target (sale price) column as y

y=pd.DataFrame(df_d.pop('SalePrice'))
# checking the shape of the train data set, to know from where to cut data set to get training data useing iloc

train.shape[0]
# using iloc on both the target and training data we can get a nice seperation between training and testing datasets

X_train = df_d.iloc[:train.shape[0] , :]

y_train = y.iloc[:train.shape[0]]
# checking train dataset shape to make sure we did correct seperation

print(X_train.shape , y_train.shape)
 # importing test/train split, and use it on training dataset to train the models and score them 

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3 , random_state = 101)

# importing scaler, then scale training and test dataframes

from sklearn.preprocessing import StandardScaler 



s = StandardScaler()



X_train_d_s = pd.DataFrame(s.fit_transform(X_train) , columns=X_train.columns)

X_test_d_s = pd.DataFrame(s.transform(X_test) , columns=X_test.columns)
#importing models 

from sklearn.linear_model import LassoCV

from sklearn.linear_model import RidgeCV

from sklearn.linear_model import ElasticNetCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
# using grid search on GradientBoostingRegressor model to get the best hyperparameters

num_estimators = [3000 , 4000]

learn_rates = [0.02, 0.05]

max_depths = [ 3 , 4]

min_samples_leaf = [10 , 15]

min_samples_split = [10 , 15]

max_features=['sqrt']



param_grid = {'n_estimators': num_estimators,

              'learning_rate': learn_rates,

              'max_depth': max_depths,

              'min_samples_leaf': min_samples_leaf,

              'min_samples_split': min_samples_split , 

             'max_features' : max_features }



grad = GridSearchCV(GradientBoostingRegressor(loss='huber'),

                           param_grid, cv=3, verbose= 1 , n_jobs=-1)

grad.fit(X_train_d_s , y_train)
# getting train score

grad.score(X_train_d_s , y_train)
# getting test scores

grad.score(X_test_d_s , y_test)
# creating dataframe containing model feature important

coef_df = pd.DataFrame({'feature': X_train_d_s.columns,

                        'importance': abs(grad.best_estimator_.feature_importances_), 

                        })



coef_df.head()
# sort by absolute value of coefficient (magnitude)

coef_df.sort_values('importance', ascending=False, inplace=True)

coef_df[:10]
# get e barplot for features

plt.figure(figsize=(7,6))

plt.xticks(rotation=60)

sns.barplot(coef_df.feature[:7] , coef_df.importance[:7],palette='RdBu') # top features
# using RandomGridSerach  to fide best hyperparametrs for RandomForestRegressor

par = {'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}



ra = RandomizedSearchCV(RandomForestRegressor(),

                   par , cv = 5 , verbose= 1  , n_jobs= -1)

ra.fit(X_train_d_s , y_train)
ra.score(X_train_d_s , y_train)
ra.score(X_test_d_s , y_test)
# creating dataframe containing model feature important

coef_df = pd.DataFrame({'feature': X_train_d_s.columns,

                        'importance': abs(ra.best_estimator_.feature_importances_), 

                        })



coef_df.head()
# sort by absolute value of coefficient (magnitude)

coef_df.sort_values('importance', ascending=False, inplace=True)

coef_df[:10]
# get e barplot for features

plt.figure(figsize=(7,6))

plt.xticks(rotation=60)

sns.barplot(coef_df.feature[:7] , coef_df.importance[:7],palette='RdBu') # top  features
# lasso with a range values for alpha

la = LassoCV(alphas=np.logspace(0, 5, 200) , n_jobs=-1)



la.fit(X_train_d_s , y_train)
la.score(X_train_d_s , y_train)
la.score(X_test_d_s , y_test)
# creating dataframe containing model feature important

coef_df = pd.DataFrame({'feature': X_train_d_s.columns,

                        'importance': abs(la.coef_),

                        })



coef_df.head()
# sort by absolute value of coefficient (magnitude)

coef_df.sort_values('importance', ascending=False, inplace=True)

coef_df[:10]
# get e barplot for features

plt.figure(figsize=(7,6))

plt.xticks(rotation=60)

sns.barplot(coef_df.feature[:7] , coef_df.importance[:7],palette='RdBu') # top features
# Ridge with a range values for alpha

ri = RidgeCV(alphas=np.logspace(0, 5, 200))



ri.fit(X_train_d_s , y_train)
ri.score(X_train_d_s , y_train)
ri.score(X_test_d_s , y_test)
# creating dataframe containing model feature important

coef_df = pd.DataFrame({'feature': X_train_d_s.columns,

                        'importance': abs(ri.coef_[0]),

                        })



coef_df.head()
# sort by absolute value of coefficient (magnitude)

coef_df.sort_values('importance', ascending=False, inplace=True)

coef_df[:10]
# get e barplot for features

plt.figure(figsize=(7,6))

plt.xticks(rotation=60)

sns.barplot(coef_df.feature[:7] , coef_df.importance[:7],palette='RdBu') # top features
# ElasticNet with a range values for alpha and l1 ratio

ela = ElasticNetCV(l1_ratio=np.arange(0.01, 1.0, 0.05) , alphas=np.logspace(0, 5, 200))



ela.fit(X_train_d_s , y_train)
ela.score(X_train_d_s , y_train)
ela.score(X_test_d_s , y_test)
# creating dataframe containing model feature important

coef_df = pd.DataFrame({'feature': X_train_d_s.columns,

                        'importance': abs(ela.coef_), # its logistic regression, we can get coef_, right?

                        })



coef_df.head()
# sort by absolute value of coefficient (magnitude)

coef_df.sort_values('importance', ascending=False, inplace=True)

coef_df[:10]
# get e barplot for features

plt.figure(figsize=(7,6))

plt.xticks(rotation=60)

sns.barplot(coef_df.feature[:7] , coef_df.importance[:7],palette='RdBu') # top features
# recreating the training and testing dataset to do the prediction on the testing data

df_d = pd.get_dummies(df , drop_first=True)

y=pd.DataFrame(df_d.pop('SalePrice'))
y_test = y.iloc[train.shape[0]:]

X_test= df_d.iloc[train.shape[0]:,:]

y_train = y.iloc[:train.shape[0]]

X_train = df_d.iloc[:train.shape[0] , :]
from sklearn.preprocessing import StandardScaler



s = StandardScaler()



X_train_d_s = pd.DataFrame(s.fit_transform(X_train) , columns=X_train.columns)

X_test_d_s = pd.DataFrame(s.transform(X_test) , columns=X_test.columns)
# creating the dataframe then save it as csv file before submiting.

sub = pd.DataFrame({

        "Id": test.Id,

        "SalePrice": grad.predict(X_test_d_s)

})



sub.head()
sub.to_csv('sub9GS.csv' , index=False)
pd.read_csv('sub9GS.csv').head()
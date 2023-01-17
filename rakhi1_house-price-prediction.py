import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Plotting

import matplotlib.pyplot as plt

import seaborn as sns

params = {'legend.fontsize': 'x-large',

          'figure.figsize': (10,8),

         'axes.labelsize': 'x-large',

         'axes.labelcolor': '#008abc',

         'axes.titlesize':'15',

         'text.color':'green',

         'axes.titlepad': 35,

         'xtick.labelsize':'small',

         'ytick.labelsize':'small'}

plt.rcParams.update(params)



# Model building & evaluation

from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.feature_selection import RFE

import statsmodels.api as sm 



# Ignore the warnings

import warnings

warnings.filterwarnings('ignore')
# Read the train data

train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

train_df.head()
# Read the test data

test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

test_df.head()
## Saving the target variable

saleprice_train = train_df.SalePrice
# Concat the train and test data for EDA puposes

housing_df = pd.concat([train_df.drop('SalePrice',axis=1),test_df])
housing_df.shape
housing_df['Id'].nunique()
housing_df.info()
## Find Percentage of NULL values

mis_val_percent = round((100 * housing_df.isnull().sum() / len(housing_df)),2)
## Fetch columns where percentage of missing records is greater than 0

mis_cols=mis_val_percent.loc[(mis_val_percent>0)].sort_values(ascending=False)

mis_cols
## Droppig columns with grater tha 45% data is missing

housing_df.drop(mis_cols[mis_cols>45].index,inplace=True,axis=1)

housing_df.info()
## Columns dropped

cols_to_drop = mis_cols[mis_cols>45].index

cols_to_drop
housing_df.drop('MiscVal',axis=1,inplace=True)
## Fetch column containing less than 45% of missing values

mis_cols[mis_cols<45]
## Fetch string datatypes

categorical_col = list(housing_df[mis_cols[mis_cols<45].index].select_dtypes(include='object').columns)
## Identify the top values and it's frequency

housing_df[categorical_col].describe()
## Function to plot the data and print the top value % contribution - for the column col passed to the function

def check_col(col):

    from IPython.display import display, HTML

    sns.countplot(housing_df[col])

    plt.title(col +": Percentage of missing value:"+str(mis_cols[col])+"%")

    plt.xticks(rotation='vertical')

    plt.show()

    text='''The top value is <b>%s</b> and it makes up for <b>%s</b> amount of the data'''%(housing_df[col].value_counts().idxmax(),round((housing_df[col].value_counts().max()/housing_df[col].count()*100),2))

    data=HTML('''<div class="alert alert-block alert-info"><span style="color:black">'''+text+'''</span></div>''')

    display(data)
## Columns related to Garage

Garage_missing = mis_cols[categorical_col[0:4]].index

Garage_missing
housing_df[Garage_missing]=housing_df[Garage_missing].replace(np.nan,'No Garage')
## Columns related to Basement

Bsmt_missing = mis_cols[categorical_col[4:9]].index

Bsmt_missing
housing_df[Bsmt_missing]=housing_df[Bsmt_missing].replace(np.nan,'No Basement')
## Masonry veneer type column - categorical_col[9]

check_col(categorical_col[9])
print(housing_df[housing_df[categorical_col[9]].isnull()]['MasVnrArea'].unique())
housing_df[categorical_col[9]]=housing_df[categorical_col[9]].replace(np.nan,'None')
## MSZoning - categorical_col[10]

check_col(categorical_col[10])
## As the evident top contributor, we will replace the missing values with RL (Residential Low density)

housing_df[categorical_col[10]]=housing_df[categorical_col[10]].replace(np.nan,'RL')
print(categorical_col)
## Utilities - categorical_col[11]

check_col(categorical_col[11])
## As the evident top contributor, we will replace the missing values with AllPub - All Public utilities

housing_df[categorical_col[11]]=housing_df[categorical_col[11]].replace(np.nan,'AllPub')
## Functional - categorical_col[12]

check_col(categorical_col[12])
## As the evident top contributor, we will replace the missing values with Typ - Typical functionality

housing_df[categorical_col[12]]=housing_df[categorical_col[12]].replace(np.nan,'Typ')
## Exterior 1st and 2nd - categorical_col[13:15]

check_col(categorical_col[13])
## Exterior1st - categorical_col[14]

check_col(categorical_col[14])
## As the evident top contributor, we will replace the missing values with VinylSd - Vinyl Siding

housing_df[categorical_col[13]]=housing_df[categorical_col[13]].replace(np.nan,'VinylSd')

housing_df[categorical_col[14]]=housing_df[categorical_col[14]].replace(np.nan,'VinylSd')
## SaleType - categorical_col[15]

check_col(categorical_col[15])
## As the evident top contributor, we will replace the missing values with WD - Warranty Deed - Conventional

housing_df[categorical_col[15]]=housing_df[categorical_col[15]].replace(np.nan,'WD')
## Electrical - categorical_col[16]

check_col(categorical_col[16])
## As the evident top contributor, we will replace the missing values with SBrkr -Standard Circuit Breakers & Romex

housing_df[categorical_col[16]]=housing_df[categorical_col[16]].replace(np.nan,'SBrkr')
## KitchenQual - categorical_col[17]

check_col(categorical_col[17])
## As the evident top contributor, we will replace the missing values with TA -Typical/Average

housing_df[categorical_col[17]]=housing_df[categorical_col[17]].replace(np.nan,'TA')
## Fetch numerical datatypes

int_mis_cols=housing_df[mis_cols[mis_cols<45].index].select_dtypes(include='float').columns

int_mis_cols
## LotFrontage

print("LotFrontage contains %s missing values"%(mis_cols.LotFrontage))
## We will impute the test and train data seperately, to avoid data leakage 

train_df['LotFrontage']=train_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

test_df['LotFrontage']=test_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
Garage_cols = int_mis_cols[int_mis_cols.str.contains('Gar')]

Basement_cols = int_mis_cols[int_mis_cols.str.contains('Bsmt')]

print(Garage_cols,"\n",Basement_cols)
## The integer variables are handled in a similar fashion like the categorical counterparts

housing_df[Garage_cols]=housing_df[Garage_cols].replace(np.nan,0)

housing_df[Basement_cols]=housing_df[Basement_cols].replace(np.nan,0)

housing_df['MasVnrArea']=housing_df['MasVnrArea'].replace(np.nan,0)
## Find Percentage of NULL values

mis_val_percent = round((100 * housing_df.isnull().sum() / len(housing_df)),2)

mis_val_percent[mis_val_percent>0]
housing_df.columns.nunique()
## Area columns with respect to SalePrice

plt.figure(figsize=(16,13))

plt.subplot(2,3,1)

plt.scatter(train_df['MasVnrArea'],train_df.SalePrice)

plt.title('MasVnrArea vs SalePrice')

plt.subplot(2,3,2)

plt.scatter(train_df['TotalBsmtSF'],train_df.SalePrice)

plt.title('TotalBsmtSF vs SalePrice')

plt.subplot(2,3,3)

plt.scatter(train_df['1stFlrSF'],train_df.SalePrice)

plt.title('1stFlrSF vs SalePrice')

plt.subplot(2,3,4)

plt.scatter(train_df['GarageArea'],train_df.SalePrice)

plt.title('GarageArea vs SalePrice')

plt.subplot(2,3,5)

plt.scatter(train_df['GrLivArea'],train_df.SalePrice)

plt.title('GrLivArea vs SalePrice')

plt.subplot(2,3,6)

plt.scatter(train_df['LotArea'],train_df.SalePrice)

plt.title('LotArea vs SalePrice')

plt.tight_layout()
sns.distplot(train_df.SalePrice)
train_df.SalePrice=np.log(train_df.SalePrice)
sns.distplot(train_df.SalePrice)
## Year as number of years from current year

import datetime

curr_year=datetime.datetime.now().year

housing_df['YearBuilt'] = curr_year - housing_df['YearBuilt']

housing_df['YearRemodAdd'] = curr_year - housing_df['YearRemodAdd']

housing_df['GarageYrBlt'] = curr_year - housing_df['GarageYrBlt']

housing_df['YrSold'] = curr_year - housing_df['YrSold']
## Determine the integer variables which are categorical in nature.

Numerics=['int64','float64']

integer_cols=housing_df.select_dtypes(include=Numerics)

integer_cols.drop('Id',axis=1,inplace=True)

int_cols = integer_cols.nunique()

int_cols[int_cols<50]
## convert integer levels to categorical type

housing_df['MSSubClass'] = housing_df['MSSubClass'].astype('object')

housing_df['OverallQual'] = housing_df['OverallQual'].astype('object')

housing_df['OverallCond'] = housing_df['OverallCond'].astype('object')

housing_df['BsmtFullBath'] = housing_df['BsmtFullBath'].astype('object')

housing_df['BsmtHalfBath'] = housing_df['BsmtHalfBath'].astype('object')

housing_df['FullBath'] = housing_df['FullBath'].astype('object')

housing_df['HalfBath'] = housing_df['HalfBath'].astype('object')

housing_df['BedroomAbvGr'] = housing_df['BedroomAbvGr'].astype('object')

housing_df['KitchenAbvGr'] = housing_df['KitchenAbvGr'].astype('object')

housing_df['TotRmsAbvGrd'] = housing_df['TotRmsAbvGrd'].astype('object')

housing_df['Fireplaces'] = housing_df['Fireplaces'].astype('object')

housing_df['GarageCars'] = housing_df['GarageCars'].astype('object')
## Check the correlation among the numerics features

Numerics=['int64','float64']

integer_cols=housing_df.select_dtypes(include=Numerics)

int_corr=integer_cols.corr()

int_corr=int_corr.transform(lambda x : round(x,2))

plt.figure(figsize=(20,20))

sns.heatmap(int_corr,cmap = plt.cm.RdYlBu_r, annot=True,vmin = -0.00,vmax = 1)
col_cat=housing_df.nunique()

binary_cols=col_cat[col_cat<3]

binary_cols.index
housing_df[binary_cols.index].apply(lambda x :print(x.name,x.unique()))
housing_df.drop(['LowQualFinSF','3SsnPorch','PoolArea'],axis=1,inplace=True)
## Perform mapping for the smaller categories

housing_df['Street']=housing_df['Street'].map({'Pave': 1, 'Grvl': 0})

housing_df['Utilities']=housing_df['Utilities'].map({'AllPub': 1,'NoSeWa':0})

housing_df['CentralAir']=housing_df['CentralAir'].map({'Y': 1, "N": 0})
## One hot encoding for the remaining categorical variables

categorical_fields=housing_df.select_dtypes(include='object')

categorical_columns=categorical_fields.nunique().sort_values(ascending=False).index

categorical_columns
dummy_vars = pd.get_dummies(housing_df[categorical_columns], drop_first=True)

dummy_vars.head()
housing_df = pd.concat([housing_df, dummy_vars], axis=1)

housing_df = housing_df.drop(categorical_columns, axis = 1)

housing_df.shape
df_train = housing_df.iloc[:1460]

df_test = housing_df.iloc[1460:]



## Fetching LotFrontage from the train and test datasets, as it was handled seperately

df_train['LotFrontage']=train_df['LotFrontage']

df_train['SalePrice'] = train_df['SalePrice']

df_test['LotFrontage']=test_df['LotFrontage']

df_test['SalePrice'] = 0
## Drop outliers for numerical columns using the Interquartile range

num_col = df_train.select_dtypes(include=Numerics).columns

num_col.drop('Id')

# num_col = ['LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF','1stFlrSF','GrLivArea','OpenPorchSF',

#            'EnclosedPorch','3SsnPorch',

#            'ScreenPorch' ,'PoolArea','MiscVal','SalePrice']

def drop_outliers(x):

    

    for col in num_col:

        Q1 = x[col].quantile(.05)

        Q3 = x[col].quantile(.95)

        IQR = Q3-Q1

        x =  x[(x[col] >= (Q1-(1.5*IQR))) & (x[col] <= (Q3+(1.5*IQR)))] 

    return x   



df_train = drop_outliers(df_train)



df_train[num_col].head()
## Area columns with respect to SalePrice

plt.figure(figsize=(16,13))

plt.subplot(2,3,1)

plt.scatter(df_train['MasVnrArea'],df_train.SalePrice)

plt.title('MasVnrArea vs SalePrice')

plt.subplot(2,3,2)

plt.scatter(df_train['TotalBsmtSF'],df_train.SalePrice)

plt.title('TotalBsmtSF vs SalePrice')

plt.subplot(2,3,3)

plt.scatter(df_train['1stFlrSF'],df_train.SalePrice)

plt.title('1stFlrSF vs SalePrice')

plt.subplot(2,3,4)

plt.scatter(df_train['GarageArea'],df_train.SalePrice)

plt.title('GarageArea vs SalePrice')

plt.subplot(2,3,5)

plt.scatter(df_train['GrLivArea'],df_train.SalePrice)

plt.title('GrLivArea vs SalePrice')

plt.subplot(2,3,6)

plt.scatter(df_train['LotArea'],df_train.SalePrice)

plt.title('LotArea vs SalePrice')

plt.tight_layout()
df_train.drop('Id',axis=1,inplace=True)
## Divide train data into X and Y set for model building



y_train = df_train.pop('SalePrice')

X_train = df_train

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train.values)

X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)



X_train.head()
# list of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 

 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 

 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100]}



ridge = Ridge()



# cross validation

folds = 5

model_cv = GridSearchCV(estimator = ridge, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            

model_cv.fit(X_train, y_train) 



## Print the best parameter lambda and the best NMSE score

print(model_cv.best_params_)

print(model_cv.best_score_)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results = cv_results[cv_results['param_alpha']<=1000]

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('int32')

plt.figure(figsize=(16,5))



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.xticks(np.arange(0,102,2))

plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper right')

plt.grid()

plt.show()
alpha = 2

ridge = Ridge(alpha=alpha)

ridge.fit(X_train, y_train)

r_coeff=ridge.coef_

r_coeff[r_coeff!=0].shape
#lets predict the R-squared value of test and train data

y_train_pred = ridge.predict(X_train)

RR2=metrics.r2_score(y_train, y_train_pred)

print("Ridge R squared (train):",RR2)
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
lasso = Lasso()



# cross validation

model_cv = GridSearchCV(estimator = lasso, 

                        param_grid = params, 

                        scoring= 'neg_mean_absolute_error', 

                        cv = folds, 

                        return_train_score=True,

                        verbose = 1)            



model_cv.fit(X_train, y_train) 
print(model_cv.best_params_)

print(model_cv.best_score_)
cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results.head()
# plotting mean test and train scoes with alpha 

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# plotting

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')

plt.xscale('log')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.grid()

plt.show()
alpha = 0.0001

lasso = Lasso(alpha=alpha)      

lasso.fit(X_train, y_train) 



lasso_c=lasso.coef_

lasso_c[lasso_c!=0].shape
#lets predict the R-squared value of test and train data

y_train_pred = lasso.predict(X_train)

LR2=metrics.r2_score(y_true=y_train, y_pred=y_train_pred)

print("Lasso R squared(Train)",LR2)
fig = plt.figure()

sns.distplot((y_train - y_train_pred), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
df_test_final = df_test.copy()

df_test.pop('Id')

df_test.head()
## Divide into X and Y set for model building



y_test = df_test['SalePrice']

X_test = df_test.loc[:, df_test.columns != 'SalePrice']

X_test_scaled = scaler.transform(X_test.values)

X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
X_test.columns.difference(X_train.columns)
#lets predict the R-squared value of test and train data using lasso

y_test_pred_lasso = np.exp(lasso.predict(X_test))

#lets predict the R-squared value of test and train data using ridge

y_test_pred_ridge = np.exp(ridge.predict(X_test))
# y_test_pred_ridge = np.exp(ridge.predict(X_test))

# y_test_pred_ridge[:10]
y_test_final = (0.2*y_test_pred_lasso+0.8*y_test_pred_ridge)
my_submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': y_test_final})

my_submission.to_csv('submission.csv', index=False)
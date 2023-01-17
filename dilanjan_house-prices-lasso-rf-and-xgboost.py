import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
#loading Data

df = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(df.shape)

print(df_test.shape)
#pd.options.display.max_columns = None

df.head(5)
df.describe()

# get only null columns

nullcol = df.columns[df.isna().any()]
df[nullcol].isnull().sum()
df[nullcol].isnull().sum() * 100 / len(df)
#dropping the columns where the missing values are over 40%



df.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
df.shape
#checking the data types of missing value columns

nullcol = df.columns[df.isna().any()]

df[nullcol].dtypes
# selecting columns where the type is strings with missing values



objcols= df[nullcol].select_dtypes(['object']).columns

objcols
#replacing the missing values of the strings with the mode

df[objcols] = df[objcols].fillna(df.mode().iloc[0])

#checking the columns

df[objcols].isnull().sum() * 100 / len(df)
#imputing numeric values



#get numerical features by dropping categorical features from the list

num_null=(nullcol.drop(objcols))



df[num_null] = df[num_null].fillna(df.mean().iloc[0])



df.columns[df.isna().any()]
#numerical data

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



num_cols = df.select_dtypes(include=numerics)



#categorical data

string_cols = df.select_dtypes(exclude =numerics)


print(num_cols.shape)

print(string_cols.shape)
#correlaation of nymerical data

corr = num_cols.corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr,cmap='coolwarm')

    


num_cols[num_cols.columns[1:]].corr()['SalePrice']
corr_matrix = num_cols.corr().abs()

high_corr_var=np.where(corr_matrix>0.8)

high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]



high_corr_var

def corrank(X):

        import itertools

        dff = pd.DataFrame([[(i,j),X.corr().loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])    

        display(dff.sort_values(by='corr',ascending=False))



corrank(num_cols) # prints a descending list of correlation pair (Max on top)
df.groupby(['YrSold','MoSold']).Id.count().plot(kind='bar',figsize=(12,4))
# this show a seasonal pattren where in 6,7 months the sales rises
chart = pd.melt( df, value_vars = num_cols )



gp = sns.FacetGrid(chart, col = 'variable', col_wrap=4, sharex=False, sharey=False)

gp = gp.map(sns.distplot,'value')

# though the above distribution is for numerical features the bars means the values

## are not continuos but discrete similar to categorcial
#print unique values for categorical columns



for cols in string_cols:

         print( cols)

         print( df[cols].unique())   

string_cols.columns

df_test.isnull().sum()
test_nullcol = df_test.columns[df_test.isna().any()]

df_test[test_nullcol].isnull().sum() * 100 / len(df_test)
#first imputing the missing values in the test data set



test_nullcol = df_test.columns[df_test.isna().any()]

df_test.isnull().sum()

df_test[test_nullcol].isnull().sum()



#string columns in test 

test_objcols= df_test[test_nullcol].select_dtypes(['object']).columns





#replacing the missing values of the strings with the mode

df_test[test_objcols] = df_test[test_objcols].fillna(df_test.mode().iloc[0])





#imputing numeric values



#get numerical features by dropping categorical features from the list

test_num_null=(test_nullcol.drop(test_objcols))



#replacing with the mean

df_test[test_num_null] = df[test_num_null].fillna(df_test.mean().iloc[0])

df_test.columns[df_test.isna().any()]
#drop features that were removed from trainig set

df_test.drop(['Alley','FireplaceQu','PoolQC','Fence','MiscFeature'], axis=1, inplace=True)
df_test.shape
#numerical data

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



#numerical features in the test set

test_num_cols = df_test.select_dtypes(include=numerics)



#categorical data features

test_string_cols = df_test.select_dtypes(exclude =numerics)
#checking whether columns match one to one

set(test_string_cols.columns.values) - set(string_cols.columns.values)

# test and training

for cols in test_string_cols:

    no = set(df_test[cols].unique() ) - set( df[cols].unique()) 

    

    if len(no) > 0:

        print(no, " ",cols)

                

# training and test



for cols in test_string_cols:

    no =  set( df[cols].unique()) -set(df_test[cols].unique() )

    

    if len(no) > 0:

        print(no, " ",cols)
for cols in test_string_cols:

    no =  set( df[cols].unique()) -set(df_test[cols].unique() )

    if len(no) > 0:

        arr=list(no)

        df.drop( df[df[cols].isin(arr)].index, inplace=True)
test_string_cols.columns
for cols in test_string_cols:

         print( cols)

         print( df_test[cols].unique())   
print(df.shape)

print(df_test.shape)
# two data sets have the same number of features (trainin has 1 more of the target )
#encoding catogorical columns

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
#encoding all categorical features



df_encoded = df[string_cols.columns].apply(le.fit_transform)
df_encoded.head(5) 
#Creating Dummy variables 



dummy = pd.get_dummies(df_encoded, columns= df_encoded.columns)

display(dummy.tail(5))
print(dummy.shape)

print(df.shape)
# merge the dummy variables to the data set 

df = pd.concat([df,dummy], axis=1)
# removing the orginal string columns since the values are represented in the dummy columns

df.drop(string_cols.columns, axis=1, inplace=True)
#removeing Id'

del df['Id']
df.head()
print("Shape of the Final Dataset", df.shape)
plt.figure(figsize=(10,6))

sns.regplot(df['GrLivArea'],df['SalePrice'] )

plt.show()
#remove outliers - houses that are more than 4000sft



df.drop(df[df.GrLivArea > 4000].index,inplace=True)
#assiging X and target label y

y = df['SalePrice']

X = df.drop('SalePrice',axis=1)
print(y.shape)

print(X.shape)
#splitting data to training and testing

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=40)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#getting numeric columns for standerdising

num_cols.drop('Id',axis=1,inplace=True)

num_cols.drop('SalePrice',axis=1,inplace=True)
num_cols.columns

X_test[num_cols.columns]
#standerdiasing

#standardising  only the numerical columns [excluding dummy varables]



from sklearn.preprocessing import StandardScaler

scale = StandardScaler()





X_train[num_cols.columns] = scale.fit_transform( X_train[num_cols.columns])

X_test[num_cols.columns] = scale.fit_transform( X_test[num_cols.columns])



print(X_test.shape)

print(X_train.shape)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
param_grid ={'alpha':[0.1,1,5,10,25],'max_iter':[50000]}

lasso = GridSearchCV(Lasso(), cv=5, param_grid=param_grid, scoring='neg_mean_squared_error')

lasso.fit(X_train, y_train)

alpha = lasso.best_params_['alpha']
alpha
#using Lasso to get important features

lmodel = Lasso(alpha=25 ).fit(X_train, y_train)



print("Train Score ",lmodel.score(X_train,y_train))

print("Test Score ",lmodel.score(X_test,y_test))



print("Number of Features Used ",np.sum(lmodel.coef_ !=0))



X_train.shape


important_f = pd.DataFrame(X.columns.values,columns =['Features'])

important_f['wht']= lmodel.coef_

important_f['Abs']= important_f['wht'].abs()

important_f = important_f.sort_values( by =['Abs'], ascending = False)



ColsUsed  = important_f[important_f['Abs'] !=0 ]

ColsUsed['Features']
#using random forest with selected features

from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators = 10, random_state = 42)

rf.fit(X_train[ColsUsed['Features']], y_train)

print("Train Score :", rf.score(X_train[ColsUsed['Features']], y_train))

print("Test  Score :",rf.score(X_test[ColsUsed['Features']], y_test))
#using gradient boosting with 161 features out of 300+

import xgboost as xgb



regr =xgb.XGBRegressor(colsample_bytree=0.2, gamma=0, learning_rate=0.05,max_depth=5,

                      min_child_weight=1,n_estimators=100,reg_alpha=0.9,

                      reg_lambda=0.6,subsample=0.2,seed=42,silent=1)
regr.fit( X_train[ColsUsed['Features']],  y_train)
print(regr.score(X_train[ColsUsed['Features']],y_train))

print(regr.score(X_test[ColsUsed['Features']],y_test))

# training score of 94 and testing 91
df_test.shape
test_string_cols
#encoding the test set categorcial columns



df_test_encoded = df_test[test_string_cols.columns].apply(le.fit_transform)

#Creating Dummy variables 



dummy = pd.get_dummies(df_test_encoded, columns= df_test_encoded.columns)
# merge the dummy variables to the data set 

df_test = pd.concat([df_test,dummy], axis=1)
print(df_test.shape)

print(dummy.shape)
# removing the orginal string columns since the values are represented in the dummy columns

df_test.drop(test_string_cols.columns, axis=1, inplace=True)
print(df_test.shape)
print(df_test.shape)
# create output dataframe with the prediction as SalesPrice and the Id for the important features



Output =pd.DataFrame(regr.predict(df_test[ColsUsed['Features']]),columns=['SalePrice'], index = df_test['Id'] )
Output.head()
Output.to_csv('Submission.csv')
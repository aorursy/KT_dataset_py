# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

pd.pandas.set_option('display.max_columns',None)
data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print(data.shape)

data.head()
#make a list of the variables that contain missing values

vars_with_na=[var for var in data.columns if data[var].isnull().sum()>1]



#print the variable name and the percentage of missing values 

for var in vars_with_na:

    print(var,np.round(data[var].isnull().mean(),3),'% missing values')
def analyse_na_value(df,var):

    df=df.copy()

    

    #Let's make a variable that indicates 1 if the observation was missing or Zero otherwise 

    df[var]=np.where(df[var].isnull(),1,0)

    

    #Let's calculate the mean SalePrice where the information is missing or present 

    df.groupby(var)['SalePrice'].median().plot.bar()

    plt.title(var)

    plt.show()

    

for var in vars_with_na:

    analyse_na_value(data,var)
# List of numerical variables

num_vars= [var for var in data.columns if data[var].dtypes!='O' ]



print('Number of numerical variables: ',len(num_vars))



# Visualise the numerical variables 

data[num_vars].head()
print('Number of House Id labels:',len(data.Id.unique()))

print('Number of Houses in the Dataset',len(data))
# List of Variables that contain year information 

year_vars=[var for var in num_vars if 'Yr' in var or 'Year' in var]

year_vars
# Lets explore the content of the years variables

for var in year_vars:

    print(var,data[var].unique())
# Evloution of House Price with Year

data.groupby('YrSold')['SalePrice'].median().plot()

plt.ylabel('Median House Price')

plt.title('Change in House Price with years');

# Lets's explore the relationship between the year variable and the house price in little more details

def analyse_year_vars(df,var):

    df=df.copy()

    

    # capture differnce between year variable and year the house was sold 

    df[var]=df['YrSold']-df[var]

    

    plt.scatter(df[var],df['SalePrice'])

    plt.ylabel('SalePrice')

    plt.xlabel(var)

    plt.show()



for var in year_vars:

    if var !='YrSold':

       analyse_year_vars(data,var)
# List of Discrete Variables 

discrete_vars=[var for var in num_vars if len(data[var].unique())<20 and var not in year_vars+['Id']]



print('Number of discrete variables:',len(discrete_vars))
# Let's visualize the discrete variables



data[discrete_vars].head()

def analyse_discrete(df,var):

    df=df.copy()

    df.groupby(var)['SalePrice'].median().plot.bar()

    plt.title(var)

    plt.ylabel('SalePrice')

    plt.show()

    

for var in discrete_vars:

    analyse_discrete(data,var)
# List of Continous variables 

cont_vars=[var for var in num_vars if var not in discrete_vars + year_vars+['Id']]



print('Number of continous variables:',len(cont_vars))
# Let's Visualize the continous variables



data[cont_vars].head()
# Lets Go ahead and analyse the distribution of this variables

def analyse_continous(df,var):

    df=df.copy()

    df[var].hist(bins=20)

    plt.ylabel('Number of houses')

    plt.xlabel(var)

    plt.title(var)

    plt.show()

    

for var in cont_vars:

    analyse_continous(data,var)

# Lets go ahead and analyse the distribution of this variables with log function

def analyse_transformed_continous(df,var):

    df=df.copy()

    

    # Log does not take negative value,so let's be careful and skip those variables 

    if 0 in data[var].unique():

        pass

    else:

        # Log transform the variable 

        df[var]=np.log(df[var])

        df[var].hist(bins=20)

        plt.ylabel('Number of houses')

        plt.xlabel(var)

        plt.title(var)

        plt.show()



for var in cont_vars:

    analyse_transformed_continous(data,var)
# Lets explore the relationship between the transformed varibales and the house Sale Price 



def transform_analyse_continous(df,var):

    df=df.copy()

    

    # Log does not take negative values, so let's be careful and skip those variables 

    if 0 in data[var].unique():

        pass

    else:

        # Log transform

        df[var]=np.log(df[var])

        df['SalePrice']=np.log(df['SalePrice'])

        plt.scatter(df[var],df['SalePrice'])

        plt.ylabel('SalePrice')

        plt.xlabel(var)

        plt.show()

        

for var in cont_vars:

    if var !='SalePrice':

        transform_analyse_continous(data,var)
# Let's make boxplots to visualise outliers in the continous variables 



def find_outliers(df,var):

    df=df.copy()

    

    # Log does not take negative values,so let's be careful and skip those variables 

    if 0 in data[var].unique():

        pass

    else:

        df[var]=np.log(df[var])

        df.boxplot(column=var)

        plt.title(var)

        plt.ylabel(var)

        plt.show()

        

for var in cont_vars:

    find_outliers(data,var)
### Categorical variables 



cat_vars=[var for var in data.columns if data[var].dtype=='O']



print('Number of categorical variables:',len(cat_vars))
# Let's visualize the values of categorical variables 

data[cat_vars].head()
for var in cat_vars:

    print(var,len(data[var].unique()),'categories')
def analyse_rare_labels(df,var,rare_perc):

    df=df.copy()

    tmp=df.groupby(var)['SalePrice'].count()/len(df)

    return tmp[tmp<rare_perc]



for var in cat_vars:

    print(analyse_rare_labels(data,var,0.01))
# to handle the dataset

import pandas as pd

import numpy as np



# for plotting 

import matplotlib.pyplot as plt

%matplotlib inline



# to divide train and test set

from sklearn.model_selection import train_test_split



# feature scaling 

from sklearn.preprocessing import MinMaxScaler



# to visualise all the columns in the dataframe

pd.pandas.set_option('display.max_columns',None)



data=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

print(data.shape)

data.head()
# Let's Separate into train and test set

# Remember to se the seed (random_state for the sklearn function)



X_train,X_test,y_train,y_test=train_test_split(data,data.SalePrice,test_size=0.1,random_state=0) # Here we are setting the seed 
#make a list of the variables that contain missing values

vars_with_na=[var for var in data.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes=='O']



#print the variable name and the percentage of missing values 

for var in vars_with_na:

    print(var,np.round(data[var].isnull().mean(),3),'% missing values')
# function to replace NA in categorical variabsables 

def fill_categorical_na(df,var_list):

    X=df.copy()

    X[var_list]=df[var_list].fillna('Missing')

    return X
# replace missing values with new label: "Missing"

X_train=fill_categorical_na(X_train,vars_with_na)

X_test=fill_categorical_na(X_test,vars_with_na)



# check that we have no missing information in the engineered variables 

X_train[vars_with_na].isnull().sum()
# check that the test set does not contain null values in the engineered variables

[vr for var in vars_with_na if X_test[var].isnull().sum()>0]
# make a list of the numerical variables that contain missing values 

vars_with_na=[var for var in data.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes!='O']



# print the variable name and the percentage of missing values 

for var in vars_with_na:

    print(var,np.round(X_train[var].isnull().mean(),3),'% missing values')
# replace the missing value 

for var in vars_with_na:

    

    # calculate the mode

    mode_val= X_train[var].mode()[0]

    

    # train

    X_train[var+'_na']= np.where(X_train[var].isnull(),1,0)

    X_train[var].fillna(mode_val,inplace=True)

    

    # test

    X_test[var+'_na']= np.where(X_test[var].isnull(),1,0)

    X_test[var].fillna(mode_val,inplace=True)

    

# check that we have no more missing values in the engineering variables 

X_train[vars_with_na].isnull().sum()
X_train[['LotFrontage_na','MasVnrArea_na','GarageYrBlt_na']].head()
# check that the test set doesnt have null values in the engineered variables 

[vr for var in vars_with_na if X_test[var].isnull().sum()>0]
# Let's explore the relationship between the year variables and the house price in bit more details



def elapsed_years(df,var):

    # capture difference between year variable and the year the house was sold

    df[var] = df['YrSold'] -df[var]

    return df
for var in ['YearBuilt','YearRemodAdd','GarageYrBlt']:

    X_train = elapsed_years(X_train,var)

    X_test = elapsed_years(X_test,var)
# check that test set does not contain null values in the engineered variables 

[vr for var in ['YearBuilt','YearRemodAdd','GarageYrBlt'] if X_test[var].isnull().sum()>0]
for var in ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']:

    X_train[var]=np.log(X_train[var])

    X_test[var]=np.log(X_test[var])   
# check that the test set does not contain null values in the engineered variables 

[var for var in ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice'] if X_test[var].isnull().sum()>0]
# check that the train set does not contain null values in the engineered variables 

[var for var in ['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice'] if X_train[var].isnull().sum()>0]
# Let's capture the categorical variables first 

cat_vars = [var for var in X_train.columns if X_train[var].dtype=='O']
def find_frequent_labels(df,var,rare_perc):

    # finds the labels that are shared by more than a certain % of the houses in the dataset 

    df=df.copy()

    tmp=df.groupby(var)['SalePrice'].count()/len(df)

    return tmp[tmp>rare_perc].index



for var in cat_vars:

    frequent_ls = find_frequent_labels(X_train,var,0.01)

    X_train[var] = np.where(X_train[var].isin(frequent_ls),X_train[var],'Rare')

    X_test[var] = np.where(X_test[var].isin(frequent_ls),X_test[var],'Rare')
# this function will assign discrete values to the strings of the variables,

# so that the similar value corresponds to the smaller mean target 



def replace_categories(train,test,var,target):

    ordered_labels=train.groupby([var])[target].mean().sort_values().index

    ordinal_label ={k:i for i,k in enumerate(ordered_labels,0)}

    train[var]=train[var].map(ordinal_label)

for var in cat_vars:

    replace_categories(X_train,X_test,var,'SalePrice')
# check absence of na

[var for var in X_train.columns if X_train[var].isnull().sum()>0]
# check absence of na

[var for var in X_test.columns if X_test[var].isnull().sum()>0]
# Let me show you what I mean by monotonic relationship between the labels and target

def analyse_vars(df,var):

    df=df.copy()

    df.groupby(var)['SalePrice'].median().plot.bar()

    plt.title(var)

    plt.ylabel('SalePrice')

    plt.show()

    

for var in cat_vars:

    analyse_vars(X_train,var)
train_vars = [var for var in X_train.columns if var not in ['Id','SalePrice']]

len(train_vars)
X_train[['Id','SalePrice']].reset_index(drop=True)
# Fit scaler 

scaler = MinMaxScaler() # create an instance 

scaler.fit(X_train[train_vars])  # fit the scaler to the train set for later user 



# transform the train and test set, and add on the Id and the SalePrice variables 

X_train = pd.concat([X_train[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scaler.transform(X_train[train_vars]),columns=train_vars)],axis=1)



X_test = pd.concat([X_test[['Id','SalePrice']].reset_index(drop=True),pd.DataFrame(scaler.transform(X_test[train_vars]),columns=train_vars)],axis=1)
X_train.head()
# Lets now save the train and test sets for the future reference

#X_train.to_csv('xtrain.csv',index=False)

#X_test.to_csv('xtest.csv',index=False)
# to build the models 

from sklearn.linear_model import Lasso 

from sklearn.feature_selection import SelectFromModel
# capture the target 

y_train = X_train['SalePrice']

y_test = X_test['SalePrice']



# drop unnecessary variables from our training and testing sets 

X_train.drop(['Id','SalePrice'],axis=1,inplace=True)

X_test.drop(['Id','SalePrice'],axis=1,inplace=True)
#

#

#



sel_ = SelectFromModel(Lasso(alpha=0.005,random_state=0))

sel_.fit(X_train,y_train)
# this command lets us visualise those feature that were kept 

# kept features are marked as True

sel_.get_support()

selected_feat=X_train.columns[(sel_.get_support())]



print('total features: {}'.format((X_train.shape[1])))

print('Selected features: {}'.format(len(selected_feat)))

print('Features with coefficients shrank to zero: {}'.format(np.sum(sel_.estimator_.coef_==0)))
selected_feat
selected_feat = X_train.columns[(sel_.estimator_.coef_!=0).ravel().tolist()]

selected_feat
#pd.Series(selected_feats).to_csv('selected_features.csv',index=False)
from sklearn.metrics import mean_squared_error

from math import sqrt
lin_model = Lasso(alpha=0.005,random_state=0)

lin_model.fit(X_train,y_train)
pred=lin_model.predict(X_train)

print('linear train mse: {}'.format(mean_squared_error(np.exp(y_train),np.exp(pred))))

print('linear train rmse:{}'.format(sqrt(mean_squared_error(np.exp(y_train),np.exp(pred)))))

print()

#pred=lin_model.predict(X_test)

#print('linear train mse: {}'.format(mean_squared_error(np.exp(y_test),np.exp(pred))))

#print('linear train rmse:{}'.format(sqrt(mean_squared_error(np.exp(y_test),np.exp(pred)))))

#print()

print('Average house price:',np.exp(y_train).median())
# Let's evaluate our predictions wrt to original price 

#plt.scatter(y_test,lin_model.predict(X_test))

#plt.xlabel('True House Price')

#plt.ylabel('Predicted House Price')

#plt.title('Evaluation of Lasso Predictions')
# Let's evaluae the distrubution of the errors :

# They should be fairly normally distributed



#errors = y_test - lin_model.predict(X_test)

#errors.hist(bins=15)
# Feature importance 



"""importance = pd.Series(np.abs(lin_model.coef_.ravel()))

importance.index = selected_feat

importance.sort_values(inplace=True,ascending=False)

importance.plot.bar(figsize=(18,6))

plt.ylabel('Lasso Coefficents')

plt.title('Feature importance')"""
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

# to visualise al the columns in the dataframe

pd.pandas.set_option('display.max_columns', None)
df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_train.head()
## Always remember there will be a chance of data leakage so we need to split the data first and then apply feature

## Engineering

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df_train,df_train['SalePrice'],test_size=0.1,random_state=0)
X_train.shape, X_test.shape
null_cat_features=[feature for feature in df_train.columns if df_train[feature].isnull().sum()>0 and df_train[feature].dtypes=='O']
## Let us capture all the nan values

## First lets handle Categorical features which are missing

def missing_cat_values(data):   

    null_cat_percentage = []

    for feature in null_cat_features:

        null_cat_percentage.append(np.round(data[feature].isnull().mean(),4)*100)

        

    data_cat_nan = pd.DataFrame({'Categorical_feature':null_cat_features,'% missing values':null_cat_percentage})

    return data_cat_nan
missing_cat_values(X_train)
missing_cat_values(X_test)
## Replace missing value with a new label

def replace_cat_feature(data):

    data_tmp = data.copy()

    data_tmp[null_cat_features] = data_tmp[null_cat_features].fillna('Missing')

    return data_tmp



X_train = replace_cat_feature(X_train)

X_test  = replace_cat_feature(X_test)
X_train[null_cat_features].isnull().sum()
print(X_test[null_cat_features].isnull().sum())
X_train.head()
X_test.head()
# lets remove Id column as it not useful for our analysis

X_train.drop('Id',axis=1,inplace=True)

X_test.drop('Id',axis=1,inplace=True)
X_train.reset_index(drop=True,inplace=True)

X_test.reset_index(drop=True,inplace=True)
X_train.head()
X_test.head()
null_num_features=[feature for feature in df_train.columns if df_train[feature].isnull().sum()>0 and df_train[feature].dtypes!='O']

## Now lets see the numerical variables which have missing values

def missing_num_values(data): 

    ## We will print the numerical nan variables and percentage of missing values

    null_num_percentage = []

    for feature in null_num_features:

        null_num_percentage.append(np.round(data[feature].isnull().mean(),4)*100)

        

    data_num_nan = pd.DataFrame({'Numerical_feature':null_num_features,'% missing values':null_num_percentage})

    return data_num_nan
missing_num_values(X_train)
missing_num_values(X_test)
## Replacing the numerical Missing Values

def replace_num_nan(data):

    for feature in null_num_features:

        ## We will replace by using training data median since there are outliers

        median_value=X_train[feature].median()

        

        ## create a new feature to capture nan values

        data[feature+'_nan']=np.where(data[feature].isnull(),1,0)

        data[feature].fillna(median_value,inplace=True)
replace_num_nan(X_train)

replace_num_nan(X_test)
missing_num_values(X_train)
missing_num_values(X_test)
year_features = ['YearBuilt','YearRemodAdd','GarageYrBlt']



X_train[year_features].head()
X_test[year_features].head()
## Temporal Variables (Date Time Variables)

def modify_year_val(data):

    for feature in year_features:

        data[feature]=data['YrSold']-data[feature]
modify_year_val(X_train)
modify_year_val(X_test)
X_train[year_features].head()
X_test[year_features].head()
num_features=['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']

X_train[num_features].head()
X_test[num_features].head()
def log_num_transform(data,num_features):

    for feature in num_features:

        data[feature]=np.log(data[feature])
log_num_transform(X_train,num_features)

log_num_transform(X_test,num_features)
X_train[num_features].head()
X_test[num_features].head()
categorical_features=[feature for feature in df_train.columns if df_train[feature].dtype=='O']
categorical_features
def modify_rare_cat(data,categorical_features):

    for feature in categorical_features:

        temp=X_train.groupby(feature)['SalePrice'].count()/len(data)

        temp_df=temp[temp>0.01].index

        data[feature]=np.where(data[feature].isin(temp_df),data[feature],'Rare_var')
modify_rare_cat(X_train,categorical_features)

modify_rare_cat(X_test,categorical_features)
def highlight_rare(s):    

    is_rare = s == 'Rare_var'

    return ['background-color: red' if v else '' for v in is_rare]
X_train[categorical_features].style.apply(highlight_rare)
X_test[categorical_features].style.apply(highlight_rare)
def label_enc_cat(data):

    for feature in categorical_features:

        labels_ordered=data.groupby([feature])['SalePrice'].mean().sort_values().index

        labels_ordered={k:i for i,k in enumerate(labels_ordered)}

        data[feature]=data[feature].map(labels_ordered)
label_enc_cat(X_train)

label_enc_cat(X_test)
X_train.head(10)
X_test.head(10)
scaling_features=[feature for feature in df_train.columns if feature not in ['Id','SalePrice','YrSold'] ]

len(scaling_features)
scaling_features
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(X_train[scaling_features])
X_train[scaling_features] = scaler.transform(X_train[scaling_features])
X_test[scaling_features] = scaler.transform(X_test[scaling_features])
X_train.head()
X_test.head()
X_train.to_csv('X_train.csv',index=False)
X_test.to_csv('X_test.csv',index=False)
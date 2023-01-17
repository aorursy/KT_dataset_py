import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
pd.set_option('display.max_columns',400)
pd.set_option('display.max_rows',300)
data_orig = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
data_orig.describe()
data_orig.info()
missing_cols = data_orig.isnull().sum().sort_values(ascending=False)
missing_cols = missing_cols[missing_cols>0].index
mask = (((data_orig[missing_cols].notnull().sum())/data_orig.shape[0])>0.05).values
drop_col = missing_cols[mask==False].tolist()
pd.DataFrame(missing_cols,columns=['Missing Columns'])
pd.DataFrame(missing_cols[mask==False],columns=['Variables'])
data_orig.drop(labels=missing_cols[mask==False],axis=1,inplace=True)
data_orig.shape[0] * 0.05    # A thresshold used to separate discrete and continous variables
def separate_variables(dataframe):            # Helper Function
    """
    Find and return discrete, continuous, catagorical and temporal variables from the given dataframe.
    
    Parameter:
    dataframe : A pandas DataFrame
    """
    year_var = []
    dis_var = []
    cat_var = []
    con_var = []
    for col in dataframe.columns:
        if dataframe[col].dtype!='O' and ('Yr' in col or 'Year' in col):
            year_var.append(col)
        elif dataframe[col].dtype=='O' and ('Yr' not in col or 'Year' not in col):
            cat_var.append(col)
        else:
            if len(dataframe[col].unique())>73:
                con_var.append(col)
            else:
                dis_var.append(col)
    return (dis_var,con_var,cat_var,year_var)
dis_var,con_var,cat_var,year_var = separate_variables(data_orig)
print("Discrete Variables : ",dis_var,end='\n\n')
print("Continuous Variables : ",con_var,end='\n\n')
print("Categorical Variables : ",cat_var,end='\n\n')
print("Temporal Variables : ",year_var,end='\n\n')
data_orig[cat_var].isnull().sum().sort_values(ascending=False)
for col in cat_var:
    data_orig[col].fillna(value=data_orig[col].mode()[0],inplace=True)
one_hot_encoded = pd.get_dummies(data_orig[cat_var],drop_first=True,dtype=float)  # One Hot Encoding
one_hot_encoded
num_data = dis_var + con_var
data_orig[num_data].isnull().sum().sort_values(ascending=False)
data_orig['LotFrontage'].fillna(value=0,inplace=True)
data_orig['LotFrontage'].where(data_orig['LotFrontage']!=0,data_orig['LotFrontage'].median(),inplace=True)
data_orig['MasVnrArea'].fillna(value=0,inplace=True)
data_orig['MasVnrArea'].where(data_orig['MasVnrArea']!=0,data_orig['MasVnrArea'].median(),inplace=True)
data_orig[dis_var + con_var].isnull().sum().sort_values(ascending=False)
data_orig[year_var].isnull().sum().sort_values(ascending=False)
data_orig['GarageYrBlt'].fillna(value=0,inplace=True)
data_orig['GarageYrBlt'].where(data_orig['GarageYrBlt']!=0,data_orig['GarageYrBlt'].median(),inplace=True)
data_orig[year_var].isnull().sum()
temp_var = ['Years_House_Old','Years_House_Remod','Years_Garage_Old']
year_dataframe = pd.DataFrame()
for col,new_col in zip(year_var[:-1],temp_var):
    year_dataframe[new_col] = data_orig['YrSold'] - data_orig[col]
for col in year_dataframe.columns:
    _,ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(year_dataframe[col],data_orig['SalePrice'],palette='coolwarm',ax=ax).set(xlabel='YearsHouseOld')
    plt.show()
fig,ax = plt.subplots(figsize=(14,7))
sns.distplot(data_orig['SalePrice'],ax=ax)
plt.show()

fig,ax = plt.subplots(figsize=(16,8))
stats.probplot(data_orig['SalePrice'],plot=ax)
plt.show()
now = np.log(data_orig['SalePrice'])
_,ax = plt.subplots(figsize=(14,7))
sns.distplot(now,ax=ax)
plt.show()

_,ax = plt.subplots(figsize=(14,7))
stats.probplot(now,plot=ax)
plt.show()
for col in dis_var:
    _,ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(col,'SalePrice',data=data_orig,palette='coolwarm',ax=ax)
    plt.show()
for col in con_var[:-1]:
    _,ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(col,'SalePrice',data=data_orig,palette='coolwarm',ax=ax)
    plt.show()
for col in cat_var:
    _,ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=col,y='SalePrice',data=data_orig,palette='coolwarm',ax=ax)
    plt.show()
def combine_dataframe(numerical_data,catagorical_data,year_data):      # Helper Function
    """
    Combine and return two given DataFrames
    numerical_data : DataFrame of shape (n_samples,n_features)
    catagorical_data : DataFrame of shape (n_samples,n_features)
    year_data : DataFrame of shape (n_samples,n_features)
    """
    dataframe = pd.concat([numerical_data,catagorical_data,year_data],axis=1,join='inner')
    return dataframe
fin_dataframe = combine_dataframe(data_orig[num_data],one_hot_encoded,year_dataframe)
fin_dataframe.drop(columns='Id',axis=1,inplace=True)
target = fin_dataframe['SalePrice'].copy()
fin_dataframe.drop(columns='SalePrice',axis=1,inplace=True)     # Dropping dependent variable
target = np.log(target)
X_train, X_test, y_train, y_test = train_test_split(fin_dataframe,target,test_size=0.30,random_state=0)
rid = Ridge(random_state=0).fit(X_train,y_train)
np.round(rid.score(X_train,y_train),decimals=2)
np.round(rid.score(X_test,y_test),decimals=2)
np.sqrt(mean_squared_error(np.exp(y_test),np.exp(rid.predict(X_test))))
param = {'alpha':[5.2,5.21]}
grid = GridSearchCV(estimator=rid,param_grid=param,n_jobs=-1)
grid.fit(X_train,y_train)
np.round(grid.score(X_train,y_train),decimals=2)
np.round(grid.score(X_test,y_test),decimals=2)
np.sqrt(mean_squared_error(np.exp(y_test),np.exp(grid.predict(X_test))))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
dataframe=pd.read_csv('../input/melbourne-housing-market/Melbourne_housing_FULL.csv')
dataframe.head()
dataframe.drop(columns=['Postcode','Lattitude','Longtitude'],inplace=True)
dataframe.shape
dataframe.isna().sum()
dataframe.dropna(axis='index',subset=['BuildingArea'],inplace=True)
dataframe.isnull().sum()
dataframe.describe().T
plt.figure(figsize=(12,9))
sns.boxplot(x=None,data=dataframe,y='BuildingArea')
dataframe=dataframe.loc[dataframe.BuildingArea>2]
dataframe.describe().T
dataframe=dataframe.loc[dataframe.BuildingArea<40000]
dataframe.describe().T
dataframe=dataframe.loc[dataframe.BuildingArea<1000]
dataframe.describe().T
plt.figure(figsize=(10,7))
sns.boxplot(y='BuildingArea',data=dataframe)
dataframe.loc[(dataframe['YearBuilt']>2018) | (dataframe['YearBuilt']<1800)]
dataframe.drop(axis=0,labels=[2453,16424,33033],inplace=True)
dataframe.describe().T
plt.figure(figsize=(10,6))
sns.boxplot(x=None,data=dataframe,y='Landsize')
plt.figure(figsize=(10,6))
sns.boxplot(x=None,data=dataframe,y='Propertycount')
dataframe=dataframe.loc[dataframe.Landsize<20000]
dataframe=dataframe.loc[dataframe.Propertycount<20000]
dataframe.describe().T
dataframe.isna().sum()
dataframe['Price'].value_counts(dropna=False)
dataframe['YearBuilt'].value_counts(dropna=False)
dataframe['Car'].value_counts(dropna=False)
dataframe.dropna(axis=0,subset=['Price'],inplace=True)
#Convert the type of some columns
dataframe['Date']=dataframe['Date'].astype('datetime64[ns]')
dataframe['Price']=dataframe['Price'].astype('int64')
dataframe['Bathroom']=dataframe['Bathroom'].astype('int64')
dataframe.isna().sum()
dataframe['Date']=dataframe['Date'].dt.year#Using only the year in the Date feature
dataframe.head(2)
dataframe['YearBuilt'].fillna(value=dataframe['YearBuilt'].mode()[0],inplace=True)
dataframe['YearBuilt']=dataframe['YearBuilt'].astype('int64')
dataframe.dtypes
dataframe['Age']=dataframe['Date']- dataframe['YearBuilt']
dataframe['TotalArea']=dataframe['Landsize']+dataframe['BuildingArea']
dataframe.head()
dataframe.describe().T
dataframe=dataframe.loc[dataframe.Age>0]
dataframe.describe().T
fig, ax=plt.subplots(figsize=(10,6))
sns.violinplot(data=dataframe,x='Regionname',y='Price',ax=ax)
ax.tick_params(labelrotation=90)
fig, ax=plt.subplots(figsize=(15,6))
sns.violinplot(data=dataframe,x='Method',y='Price',ax=ax)#Conocer la relacion existente entre El method y el precio
ax.tick_params(labelrotation=90)
fig, ax=plt.subplots(figsize=(15,6))
sns.violinplot(data=dataframe,x='Type',y='Price',ax=ax)
dataframe.drop(columns=['Suburb','Address','SellerG','Method','Landsize',
                        'BuildingArea','Date','YearBuilt','CouncilArea'],inplace=True)
dataframe.head()
corr=dataframe.corr()
fig, ax= plt.subplots(figsize=(15,10))
sns.heatmap(data=corr,annot=True,cmap='coolwarm')
dataframe.drop(columns=['Bedroom2'],inplace=True)
dataframe.head()
X=dataframe.drop(columns=['Price'],axis='columns')
y= dataframe[['Price']]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25, random_state=42)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
si=SimpleImputer(strategy='most_frequent',missing_values=np.nan)
num_col=X.select_dtypes(exclude='object').columns
cat_col=X.select_dtypes(include='object').columns
ohe=OneHotEncoder(drop='first')
num_transform=Pipeline([('si',si),('ss',StandardScaler())])
cat_transform=Pipeline([('ohe',ohe)])
ct=ColumnTransformer(transformers=[('num',num_transform,num_col),
                                   ('cat',cat_transform,cat_col)])
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
modelo=Pipeline([('ct',ct),
                 ('lr',LinearRegression())])
from sklearn.model_selection import cross_validate
scores=cross_validate(modelo,X_train,y_train,cv=5,scoring='neg_mean_absolute_error',return_train_score=True)
print(f"MAE of Training {-1*scores['train_score'].mean()}")
print(f"MAE of Validation {-1*scores['test_score'].mean()}")
from sklearn.ensemble import GradientBoostingRegressor
ct7=ColumnTransformer(transformers=[('num',num_transform,num_col),
                                 ('cat',cat_transform,cat_col)])
model7=Pipeline([('ct',ct7),
                 ('gbr',GradientBoostingRegressor(criterion='mae',random_state=0,learning_rate=0.1,
                                                 max_depth=6,min_samples_leaf=3,n_estimators=100))])
scores2=cross_validate(model7,X_train,y_train,cv=5,scoring='neg_mean_absolute_error',return_train_score=True)
print(f"MAE of Training {-1*scores2['train_score'].mean()}")
print(f"MAE of Validation {-1*scores2['test_score'].mean()}")
model7.fit(X_train,y_train)
from sklearn import metrics
y_pred_train=model7.predict(X_train)
y_pred_test=model7.predict(X_test)
print(f'MAE train {metrics.mean_absolute_error(y_train,y_pred_train)}')
print(f'MAE test {metrics.mean_absolute_error(y_test,y_pred_test)}')
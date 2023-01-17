# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
plt.style.use('fivethirtyeight')
train = pd.read_csv('/kaggle/input/bigmart-sales-data/Train.csv')
test = pd.read_csv('/kaggle/input/bigmart-sales-data/Test.csv')
train.head()
shape = train.shape
print('The train data had {} rows and {} columns'.format(shape[0],shape[1]))
train.info()
for i,j in enumerate(train.columns):
    print(i,j)
train.duplicated().sum()
def missing_value(df):
    nan = df.isnull().sum()
    nan_proportion = (nan/len(df))*100
    nan_table = pd.concat([nan,nan_proportion],axis=1).rename(columns = {0:'missing values',1:'precent of missing values'})
    return nan_table
missing_value(train)
for col in train.columns:
    print('Column' ,col,'had', train[col].nunique(),'unique value and its type is',train[col].dtype)
train['Item_Fat_Content'].unique()
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Weight'].mode()[0])
train['Outlet_Size'] = train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0])
assert train['Item_Weight'].isnull().sum() == 0
assert train['Outlet_Size'].isnull().sum() == 0
missing_value(train)
train['Item_Fat_Content']=train['Item_Fat_Content'].replace('reg','Regular')
train['Item_Fat_Content']=train['Item_Fat_Content'].replace(['low fat','LF'],'Low Fat')
assert train['Item_Fat_Content'].nunique() == 2
train['Item_Fat_Content'].unique()
train['Item_Identifier']=train['Item_Identifier'].apply(lambda x:x[0:2])
train['Item_Identifier'].unique()
train[['Item_Identifier','Item_Fat_Content' ,'Item_Type','Outlet_Identifier' ,'Outlet_Size','Outlet_Location_Type','Outlet_Type']]=train[['Item_Identifier','Item_Fat_Content' ,'Item_Type','Outlet_Identifier' ,'Outlet_Size','Outlet_Location_Type','Outlet_Type']].astype('category')
train[['Item_Identifier','Item_Fat_Content' ,'Item_Type','Outlet_Identifier' ,'Outlet_Size','Outlet_Location_Type','Outlet_Type']].info()
train.head()
train.info()
test.head()
shape = test.shape
print('The test data had {} rows and {} columns'.format(shape[0],shape[1]))
test.info()
for i,j in enumerate(test.columns):
    print(i,j)
test.duplicated().sum()
missing_value(test)
for col in test.columns:
    print('Column' ,col,'had', test[col].nunique(),'unique value and its type is',test[col].dtype)
test['Item_Fat_Content'].unique()
test['Item_Weight'] = test['Item_Weight'].fillna(test['Item_Weight'].mode()[0])
test['Outlet_Size'] = test['Outlet_Size'].fillna(test['Outlet_Size'].mode()[0])
assert test['Item_Weight'].isnull().sum() == 0
assert test['Outlet_Size'].isnull().sum() == 0
missing_value(test)
test['Item_Fat_Content']=test['Item_Fat_Content'].replace('reg','Regular')
test['Item_Fat_Content']=test['Item_Fat_Content'].replace(['low fat','LF'],'Low Fat')
assert test['Item_Fat_Content'].nunique() == 2
test['Item_Fat_Content'].unique()
test['Item_Identifier']=test['Item_Identifier'].apply(lambda x:x[0:2])
test['Item_Identifier'].unique()
test[['Item_Identifier','Item_Fat_Content' ,'Item_Type','Outlet_Identifier' ,'Outlet_Size','Outlet_Location_Type','Outlet_Type']]=test[['Item_Identifier','Item_Fat_Content' ,'Item_Type','Outlet_Identifier' ,'Outlet_Size','Outlet_Location_Type','Outlet_Type']].astype('category')
test[['Item_Identifier','Item_Fat_Content' ,'Item_Type','Outlet_Identifier' ,'Outlet_Size','Outlet_Location_Type','Outlet_Type']].info()
test.head()
test.info()
df = pd.concat([train,test])
df.head()
df.info()
df.describe()
plt.figure(figsize=(12,6))
plt.hist(df.Item_Weight,bins=30)
plt.xlabel('Item Weight')
plt.ylabel('Count')
plt.title('Graph of Item Weight')
plt.show()
plt.figure(figsize=(12,6))
sns.violinplot(x=df.Item_Weight,bins=30)
plt.xlabel('Item Weight')
plt.title('Graph of Item Weight')
plt.show()
plt.figure(figsize=(12,6))
plt.pie(df.Item_Fat_Content.value_counts(),explode=[0.1,0.1],labels=['Low Fat','Regular'],autopct='%.1f%%',shadow=True)
plt.axis('equal')
plt.title('Graph of Item Fat Content')
plt.show()
df.Item_Fat_Content.value_counts()
plt.figure(figsize=(12,6))
plt.hist(df.Item_Visibility,bins=20,color='red')
plt.xlabel('Item Visibility')
plt.ylabel('Count')
plt.title('Graph of Item Visibility')
plt.show()
plt.figure(figsize=(12,6))
sns.boxplot(df.Item_Visibility,color='red')
plt.xlabel('Item Visibility')
plt.title('Graph of Item Visibility')
plt.show()
#initializing plot
ax = df.Item_Type.value_counts().plot.barh(color = '#007482', fontsize = 15)

#giving a title
ax.set(title = 'Graph of Item Type')

#x-label
ax.set_xlabel('Item Type', color = 'g', fontsize = '18')

#giving the figure size(width, height)
ax.figure.set_size_inches(12, 10)

#shwoing the plot
plt.show()
df.Item_Type.value_counts(normalize=True)
plt.figure(figsize=(12,6))
sns.violinplot(df.Item_MRP,bins=20,color='orange')
plt.xlabel('Item MRP')
plt.title('Graph of Item MRP')
plt.show()
#initializing plot
ax = df.Outlet_Establishment_Year.value_counts().sort_index().plot.bar(color = '#007482', fontsize = 15)

#giving a title
ax.set(title = 'Graph of Outlet Establishment Year')

#x-label
ax.set_xlabel('Year', color = 'g', fontsize = '18')

#giving the figure size(width, height)
ax.figure.set_size_inches(12, 10)

#shwoing the plot
plt.show()
plt.figure(figsize=(12,6))
plt.pie(df.Outlet_Size.value_counts(),explode=[0.1,0.1,0.1],labels=['Medium', 'small', 'High'],autopct='%.1f%%',shadow=True)
plt.axis('equal')
plt.title('Graph of Outlet Size')
plt.show()
df.Outlet_Size.value_counts()
plt.figure(figsize=(12,6))
plt.pie(df.Outlet_Location_Type.value_counts(),explode=[0.1,0.1,0.1],labels=['Tier 3', 'Tier 2', 'Tier 1'],autopct='%.1f%%',shadow=True)
plt.axis('equal')
plt.title('Graph of Outlet Location Type')
plt.show()
df.Outlet_Location_Type.value_counts()
#initializing plot
ax = df.Outlet_Type.value_counts().plot.barh(color = '#007482', fontsize = 15)

#giving a title
ax.set(title = 'Graph of Outlet Type')

#x-label
ax.set_xlabel('Outlet Type', color = 'g', fontsize = '18')

#giving the figure size(width, height)
ax.figure.set_size_inches(12, 10)

#shwoing the plot
plt.show()
df.Outlet_Type.value_counts(normalize=True)
plt.figure(figsize=(12,6))
plt.hist(df.Item_Outlet_Sales,bins=30)
plt.xlabel('Item Outlet Sales')
plt.ylabel('Count')
plt.title('Graph of Item Outlet Sales')
plt.show()
df_outlier = train.Item_Outlet_Sales
df_outlier_Q1 = df_outlier.quantile(0.25)
df_outlier_Q3 = df_outlier.quantile(0.75)
df_outlier_IQR = df_outlier_Q3 - df_outlier_Q1
df_outlier_lower = df_outlier_Q1 - (1.5 * df_outlier_IQR)
df_outlier_upper = df_outlier_Q3 + (1.5 * df_outlier_IQR)
(df_outlier_lower,df_outlier_upper)
index = train.query('Item_Outlet_Sales >= 6501.8699 or Item_Outlet_Sales <= -2566.3261').index
train.drop(index,inplace=True)
train = train.reset_index(drop=True)
train['Item_Identifier'] = train['Item_Identifier'].cat.codes
train['Item_Fat_Content'] = train['Item_Fat_Content'].cat.codes
train['Item_Type'] = train['Item_Type'].cat.codes
train['Outlet_Identifier'] = train['Outlet_Identifier'].cat.codes
train['Outlet_Size'] = train['Outlet_Size'].cat.codes
train['Outlet_Location_Type'] = train['Outlet_Location_Type'].cat.codes
train['Outlet_Type'] = train['Outlet_Type'].cat.codes
test['Item_Identifier'] = test['Item_Identifier'].cat.codes
test['Item_Fat_Content'] = test['Item_Fat_Content'].cat.codes
test['Item_Type'] = test['Item_Type'].cat.codes
test['Outlet_Identifier'] = test['Outlet_Identifier'].cat.codes
test['Outlet_Size'] = test['Outlet_Size'].cat.codes
test['Outlet_Location_Type'] = test['Outlet_Location_Type'].cat.codes
test['Outlet_Type'] = test['Outlet_Type'].cat.codes
plt.figure(figsize=(15,12))
sns.heatmap(train.corr(),annot=True,cmap='Blues')
train.head()
from sklearn.model_selection import train_test_split
X = train.drop('Item_Outlet_Sales',axis=1)
y = train['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.intercept_
lm.coef_
predict = lm.predict(X_test)
plt.figure(figsize=(12,6))
plt.scatter(y_test,predict,color='grey')
plt.ylabel('predict')
plt.xlabel('y_test')
plt.show()
from sklearn.metrics import mean_squared_error,r2_score
RMSE  = np.sqrt(mean_squared_error(y_test,predict))
RMSE 
test_predict = lm.predict(test)
test['Item_Outlet_Sales'] = test_predict
test
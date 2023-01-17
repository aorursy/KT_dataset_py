import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train=pd.read_csv('../input/black-friday/train.csv')

test=pd.read_csv('../input/black-friday/test.csv')
train.head()
test.head()
print('train:',train.shape)

print('test:',test.shape)
df=pd.concat([train,test],axis=0,sort=False,ignore_index=True)

df.head()
df.shape
train.info()
train.describe()
train.nunique()
(df.isnull().sum()*100/df.shape[0]).sort_values(ascending=False)
#Since 69.6% of data is null in Product_Category_3 we will remove that column

df=df.drop('Product_Category_3',axis=1)

train=train.drop('Product_Category_3',axis=1)

test=test.drop('Product_Category_3',axis=1)
sns.kdeplot(df['Product_Category_2'])

plt.show()
vc = df['Product_Category_2'].value_counts(normalize = True)

null = df['Product_Category_2'].isnull()

df.loc[null, 'Product_Category_2'] = np.random.choice(vc.index, size = null.sum(), p = vc.values)
# Checking if all the null values have been removed

df['Product_Category_2'].isnull().sum()
# Showing that the distribution have not changed

sns.kdeplot(df['Product_Category_2'])

plt.show()
a=pd.crosstab(train['Age'],train['User_ID'])

a
# The table shows which customer from each age group who  have purchased the maximum times.

b=pd.DataFrame()

b['Maximum Purchase Count']=a.max(axis=1).values

b['User_ID']=a.idxmax(axis=1).values

b.index=a.index

b
print('Customer',a.max(axis=0).idxmax(),'of age between',a.max(axis=1).idxmax(),'have purchased maximum number of times.' )
c=pd.crosstab(train['City_Category'],train['Product_ID'])

c
d=pd.DataFrame()

d['Maximum Purchase Count']=c.max(axis=1).values

d['Product_ID']=c.idxmax(axis=1).values

d.index=c.index

d
# The popular product in each age group along with their count

a=pd.crosstab(train['Age'],train['Product_ID'])

b=pd.DataFrame()

b['Maximum Purchase Count']=a.max(axis=1).values

b['Product_ID']=a.idxmax(axis=1).values

b.index=a.index

b
# The popular product according to Marital_Status along with their count

a=pd.crosstab(train['Marital_Status'],train['Product_ID'])

b=pd.DataFrame()

b['Maximum Purchase Count']=a.max(axis=1).values

b['Product_ID']=a.idxmax(axis=1).values

b.index=a.index

print(b)
# Maximum and Minimum amount spent by a person

train.groupby('User_ID').sum()['Purchase'].sort_values()
# Creating a data frame to summarize the results

min_max=pd.DataFrame(columns=['Gender', 'Age', 'Occupation', 'City_Category',

       'Stay_In_Current_City_Years', 'Marital_Status'],index=['Minimum','Maximum'])
# Gender 

print(train.groupby('Gender').mean()['Purchase'].sort_values())

min_max['Gender']=['F','M']

# Age 

print(train.groupby('Age').mean()['Purchase'].sort_values())

min_max['Age']=['0-17','51-55']

# Occupation

print(train.groupby('Occupation').mean()['Purchase'].sort_values())

min_max['Occupation']=['9','17']

#City_Category

print(train.groupby('City_Category').mean()['Purchase'].sort_values())

min_max['City_Category']=['A','C']

#Stay_In_Current_City_Years

print(train.groupby('Stay_In_Current_City_Years').mean()['Purchase'].sort_values())

min_max['Stay_In_Current_City_Years']=['0','2']
# Marital_Status

print(train.groupby('Marital_Status').mean()['Purchase'].sort_values())

min_max['Marital_Status']=['1','0']
min_max
fig, axes = plt.subplots(2, 3, figsize = (15,8))

axes = axes.flatten()



for i in range(0,len(train.columns)-5):

    sns.countplot(train.iloc[:,i+2], data=train, ax=axes[i])

plt.show()
plt.figure(figsize=(10,4))

sns.heatmap(train.corr(),annot=True)

plt.show()
# Product category 1 and 2 seem to have a bit correlation.
# Visual representation of what Product_Category_1, Product_Category_2 are prefered with respect to Gender, Marital status and city

fig,ax =plt.subplots(3,2,figsize=(15,15))

ax=ax.flatten()



pd.crosstab(train['Product_Category_1'],train['Gender']).plot(kind='bar',stacked=True,ax=ax[0])

pd.crosstab(train['Product_Category_2'],train['Gender']).plot(kind='bar',stacked=True,ax=ax[1])



pd.crosstab(train['Product_Category_1'],train['Marital_Status']).plot(kind='bar',stacked=True,ax=ax[2])

pd.crosstab(train['Product_Category_2'],train['Marital_Status']).plot(kind='bar',stacked=True,ax=ax[3])



pd.crosstab(train['Product_Category_1'],train['City_Category']).plot(kind='bar',stacked=True,ax=ax[4])

pd.crosstab(train['Product_Category_2'],train['City_Category']).plot(kind='bar',stacked=True,ax=ax[5])



plt.show()
train.columns
# We will consider Stay_In_Current_City_Years =4+ as just 4.

df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].replace('4+','4')

df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype('int64')
# Since age can be considered as ordinal we use label encoding

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df['LE Age']=le.fit_transform(df.Age)
# Shows which label is given to which Age group

compare=pd.DataFrame(columns=['Label','Age'])

compare['Label']=df['LE Age'].value_counts().index

compare['Age']=df['Age'].value_counts().index

compare.sort_values(by='Label')
df=df.drop('Age',axis=1)
# Removing the the prefix 'P' from Product ID and converting it to an integer

df['Product_ID']=df['Product_ID'].str.lstrip('P').astype('int64')
# Changing the rest of the categorical data into numerical data.

df=pd.get_dummies(df,drop_first=True)

df.head()
# Splitting the data into train and test

df_train=df[0:550068]

df_test=df[550068:783668]
df_test=df_test.drop('Purchase',axis=1)
X=df_train.drop('Purchase',axis=1)

y=df_train['Purchase']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)



from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

X_train_std=pd.DataFrame(ss.fit_transform(X_train),columns=X_train.columns)

X_test_std=pd.DataFrame(ss.transform(X_test),columns=X_test.columns)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error



lr=LinearRegression()

lr.fit(X_train_std,y_train)

y_pred_test=lr.predict(X_test_std)





print('R^2 on the test data', r2_score(y_test, y_pred_test))

print('RMSE on the test data', np.sqrt(mean_squared_error(y_test, y_pred_test)))
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=10, min_samples_leaf=500)

dt.fit(X_train, y_train)

y_pred_test=dt.predict(X_test)



print('R^2 on the test data', r2_score(y_test, y_pred_test))

print('RMSE on the test data', np.sqrt(mean_squared_error(y_test, y_pred_test)))
from sklearn.ensemble import RandomForestRegressor



rf = RandomForestRegressor(n_estimators=30,random_state=3,max_depth=15,min_samples_split=100)

rf.fit(X_train,y_train)

y_pred_test=rf.predict(X_test)



print('R^2 on the test data', r2_score(y_test, y_pred_test))

print('RMSE on the test data', np.sqrt(mean_squared_error(y_test, y_pred_test)))
from xgboost import XGBRegressor

from sklearn.preprocessing import StandardScaler



xgb= XGBRegressor(learning_rate=0.05,n_estimators=500,max_depth=10)

xgb.fit(X_train,y_train)



y_pred_test=xgb.predict(X_test)

print('R^2 on the test data', r2_score(y_test, y_pred_test))

print('RMSE on the test data', np.sqrt(mean_squared_error(y_test, y_pred_test)))

from sklearn.ensemble import GradientBoostingRegressor



GBoost = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05,

                                   min_samples_leaf=500, min_samples_split=100)



GBoost.fit(X_train,y_train)

y_pred_test=GBoost.predict(X_test)



print('R^2 on the test data', r2_score(y_test, y_pred_test))

print('RMSE on the test data', np.sqrt(mean_squared_error(y_test, y_pred_test)))

import lightgbm as lgb

LightGB = lgb.LGBMRegressor(objective='regression',num_leaves=500,

                              learning_rate=0.05, n_estimators=100,

                              bagging_fraction = 0.8,bagging_freq = 5,                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf = 100)



LightGB.fit(X_train,y_train)

y_pred_test=LightGB.predict(X_test)



print('R^2 on the test data', r2_score(y_test, y_pred_test))

print('RMSE on the test data', np.sqrt(mean_squared_error(y_test, y_pred_test)))

from sklearn.ensemble import VotingRegressor



VR=VotingRegressor(estimators=[('xgb',xgb),('rf',rf),('LightGB',LightGB)],weights=[5,1,1])

VR.fit(X_train,y_train)



y_pred_test=VR.predict(X_test)



print('R^2 on the test data', r2_score(y_test, y_pred_test))

print('RMSE on the test data', np.sqrt(mean_squared_error(y_test, y_pred_test)))

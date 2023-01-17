import pandas as pd
import numpy as np
%%time
df_train=pd.read_csv('../input/train.csv')
%%time
df_test=pd.read_csv('../input/test.csv')
df_train.info()
df_train.describe() ##returns only numeric value
df_test.describe()
df_train.describe(include='all')  ##returns all numeric and categorical features
df_train.head()
df_train.isnull().sum()  ##age,cabin,embarked has null values
df_train.head()
print(df_train.shape)
print(df_test.shape)
df_test.head()
df_train['Survived'].unique()
df_train.columns
df_train=df_train.drop(['Ticket'],axis=1)
df_train=df_train.drop(['Name'],axis=1)
df_train=df_train.drop(['PassengerId'],axis=1)
df_train.columns
df_train.head()
df_train.tail()
mean_value=df_train['Age'].mean()
print(mean_value)
df_train['Age']=df_train['Age'].fillna(int(mean_value))
print(df_train.head())
print(df_train.sample(6))
print(df_train.tail())
df_train.isnull().sum()
df_train['Embarked']=df_train['Embarked'].astype('category') 
df_train['Sex']=df_train['Sex'].astype('category')
df_train.dtypes
df_train=df_train.drop(['Cabin'],axis=1)
df_train.dtypes
cat_columns=df_train.select_dtypes(['category']).columns
df_train[cat_columns]=df_train[cat_columns].apply(lambda x: x.cat.codes)
df_train.head()
df_train.dtypes
df_train[cat_columns].sample(5)
print(df_test.shape)
df_test.head()
df_test=df_test.drop(['Ticket'],axis=1)
df_test=df_test.drop(['Name'],axis=1)
df_test=df_test.drop(['Cabin'],axis=1)
df_test.head()
df_test.isnull().sum()
#del df_test
mean_val=df_test['Age'].mean()
print(mean_val)
df_test['Age']=df_test['Age'].fillna(int(mean_val))

print(df_test.head())
print(df_test.sample(6))
print(df_test.tail())
df_test.head()
df_test['Embarked'] = df_test['Embarked'].astype('category')
df_test['Sex'] = df_test['Sex'].astype('category')
 
df_test.dtypes
df_test.sample(4)
cat_col=df_test.select_dtypes(['category']).columns
df_test[cat_col]=df_test[cat_col].apply(lambda x: x.cat.codes)
df_test.head()
df_test.isnull().sum()
df_test.sample(10)
df_test=df_test.replace(np.nan,1)
df_test.isnull().sum()
df_train.head()
x_train=df_train.iloc[:,1:8]
print(x_train.head())
print(x_train.shape)
y_train=df_train.iloc[:,0:1]
y_train.head()
df_test.shape
df_test.head()
df=pd.DataFrame()
df=df_test.iloc[:,0:1]
df.head()
df_test=df_test.drop(['PassengerId'],axis=1)
df_test.shape
x_test=df_test.iloc[:,:]
x_test.head()
%%time
from sklearn.ensemble import RandomForestClassifier 
rf= RandomForestClassifier(random_state=0,n_estimators=10)
rf
%%time
rf.fit(x_train,y_train)
%%time
print(rf.score(x_train,y_train))
%%time
x_test_result=rf.predict(x_test)
x_test_result
x_test_df=pd.DataFrame(x_test_result)
x_test_df=x_test_df.rename({0:"Survived"},axis=1)
# df.rename({1: 2, 2: 4}, axis='index')
x_test_df.sample(4)
df_test.columns
x_test_df1=df_test.iloc[:,0:1]
df_test.sample(2)
x_test_df1.sample(3)
x_test_final=pd.concat([x_test_df1,x_test_df],axis=1)
x_test_final.head()
x_test_final.to_csv('gender_submission.gz',index=False,compression='gzip')
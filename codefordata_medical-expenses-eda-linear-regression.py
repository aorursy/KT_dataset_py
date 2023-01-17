import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/insurance.csv')
df.head()
df.info()
df.isnull().sum()
sns.heatmap(df.isnull(),cmap="YlGnBu")
df.dtypes
df[['sex','smoker','region']].head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

le.fit(df.sex.drop_duplicates())

df.sex = le.transform(df.sex)
df['sex'].head()
le.fit(df.smoker.drop_duplicates())
df.smoker=le.transform(df.smoker)
le.fit(df.region.drop_duplicates())
df.region = le.transform(df.region)
df.head()
df.dtypes
df.corr()
sns.heatmap(df.corr(),cmap="YlGnBu")
corr_matrix = df.corr()
corr_matrix['charges'].sort_values(ascending=False)
sns.countplot(data=df,x='smoker',hue='sex')
sns.countplot(data=df,x='smoker')
sns.countplot(data=df,x='region')
sns.countplot(data=df,x='sex')
sns.countplot(data=df,x='smoker',hue='sex')
df['charges'].hist()
df['charges'].min()
df.charges.max()
f = plt.figure(figsize=(12,7))

ax = f.add_subplot(121)

sns.distplot(df[df['smoker']==1]['charges'],color='g',ax=ax)

plt.title('Distrubution of  charges for Smokers')



ax = f.add_subplot(122)

sns.distplot(df[df['smoker']==0]['charges'],color='b',ax=ax)

plt.title('Distrubution of  charges for Non-Smokers')
df[df['smoker']==0]['charges'].max()
df[df['smoker']==0]['charges'].min()
df[df['smoker']==1]['charges'].min()
df[df['smoker']==1]['charges'].max()
sns.catplot(x='sex',y='charges',hue='smoker',kind='violin',data=df)
plt.figure(figsize=(12,5))

plt.title("Box plot for charges of women")

sns.boxplot(y="smoker", x="charges", data =  df[(df.sex == 1)] , orient="h", palette = 'magma')
plt.figure(figsize=(12,5))

plt.title("Box plot for charges of women")

sns.boxplot(y="smoker", x="charges", data =  df[(df.sex == 0)] , orient="h", palette = 'magma')
df.age.plot(kind='hist')
df['age'].min()
df['age'].max()
plt.figure(figsize=(12,8))

sns.catplot(x='smoker',kind='count',hue='sex',data=df[(df.age<=23)])
df[(df['smoker']==0) & (df['age']<=23) &(df['sex']==0)].count()


sns.jointplot(x="age", y="charges", data = df[(df.smoker == 0)],kind="kde", color="m")
sns.jointplot(x="age", y="charges", data = df[(df.smoker == 1)],kind="kde", color="g")
sns.distplot(df['bmi'])
plt.figure(figsize=(14,8))

plt.title('Distribution of charges for patients with BMI greater than 30')

ax = sns.distplot(df[df.bmi>=30]['charges'],color='m')
plt.figure(figsize=(14,8))

plt.title('Distribution of charges for patients with BMI lesser than 30')

ax = sns.distplot(df[df.bmi<30]['charges'],color='g')
sns.jointplot(x="bmi", y="charges", data = df,kind="kde", color="r")
sns.catplot(x="children", kind="count", data=df, size = 6)
sns.catplot(x="children", kind="count", hue='smoker',data=df, size = 6)
X = df.drop(['charges'],axis=1)

y= df['charges']
X.head()
y.head()
print(X.shape)

print(y.shape)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
l_reg = LinearRegression()
l_reg.fit(X_train,y_train)
y_pred = l_reg.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))
print(l_reg.score(X_test,y_test))
X_train.head()
y_train.head()
l_reg.predict([[48,1,35.625,4,0,0]])
corr_matrix['charges']
X = df.drop(['region','charges'],axis=1)  
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
l_reg.fit(X_train,y_train)
y_pred = l_reg.predict(X_test)
l_reg.score(X_test,y_test)
mean_squared_error(y_test,y_pred)
print(r2_score(y_test,y_pred))
X_train.head()
y_train.head()
l_reg.predict([[18,0,38.280,0,0]])
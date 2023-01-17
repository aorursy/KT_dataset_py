import pandas as pd
data=pd.read_csv('../input/insurance/insurance.csv')
data.shape
data.head()
# check for null values



data.isnull().sum()
data['sex'].unique()
data['smoker'].unique()
data['region'].unique()
from sklearn.preprocessing import LabelEncoder

label_encoder=LabelEncoder()
data.sex=label_encoder.fit_transform(data.sex)

data.smoker=label_encoder.fit_transform(data.smoker)

data.region=label_encoder.fit_transform(data.region)
data.head()
data.info()
data.shape
data.describe()
data.head()
y=data['charges']

X=data.drop(['charges'],axis=1)
from sklearn.model_selection import train_test_split 

from sklearn import linear_model
reg = linear_model.LinearRegression() 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
reg.fit(X_train, y_train) 
reg.score(X_test,y_test)
data.corr()['charges'].sort_values()
X_new=X.drop(['region'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.20)
reg_1 = linear_model.LinearRegression() 
reg_1.fit(X_train, y_train) 
reg_1.score(X_test,y_test)
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data_1=pd.read_csv('../input/insurance/insurance.csv')
data_1.head()
correlation=data.corr()
correlation
sns.heatmap(correlation)
# no of male and females in the complete dataset



sns.countplot(data_1['sex']);
data_1.head()
data_1['charges'].describe()
data_1['age'].describe()
#data of people who are between 18 and 25



data_u25=data_1[data_1['age']<=25]
data_u25.head()
# people who are between 26 and 34



data_u34=data_1.loc[(data_1['age']>=26) & (data_1['age']<=34)]
data_u34.head()
# between 35 and 49



data_u49=data_1.loc[(data_1['age']>=35) & (data_1['age']<=49)]
data_u49.head()
# people who are between 50 and 64



data_u64=data_1.loc[(data_1['age']>=50) & (data_1['age']<=64)]
data_u64.head()
def shape(df):

    print(df.shape)
shape(data_1)

shape(data_u25)

shape(data_u34)

shape(data_u49)

shape(data_u64)
def scatter_plot(x,y,x_heading,y_heading) :  

    plt.scatter(x=x,y=y)

    plt.xlabel(x_heading)

    plt.ylabel(y_heading)
#relation between BMI and CHARGES





plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

scatter_plot(data_u25['bmi'],data_u25['charges'],'BMI of people between 18 and 25','CHARGES')

plt.subplot(2,2,2)

scatter_plot(data_u34['bmi'],data_u34['charges'],'BMI of people between 26 and 34 ','CHARGES')

plt.subplot(2,2,3)

scatter_plot(data_u49['bmi'],data_u49['charges'],'BMI of people between 35 and 49 ','CHARGES')

plt.subplot(2,2,4)

scatter_plot(data_u64['bmi'],data_u64['charges'],'BMI of people between 50 and 64 ','CHARGES')
def boxplot(x,y):

    sns.boxplot(x=x,y=y)
#medical charges according to age 





plt.figure(figsize=(12,12))

plt.subplot(2,2,1)

boxplot(x=data_u25['age'],y=data_u25['charges']) 

plt.xlabel('Age between 18 and 25')

plt.subplot(2,2,2)

boxplot(x=data_u34['age'],y=data_u34['charges']) 

plt.xlabel('Age between 26 and 34')

plt.subplot(2,2,3)

boxplot(x=data_u49['age'],y=data_u49['charges']) 

plt.xlabel('Age between 35 and 49')

plt.subplot(2,2,4)

boxplot(x=data_u64['age'],y=data_u64['charges']) 

plt.xlabel('Age between 50 and 64');
#medical charges for people who smoke vs who don't smoke





plt.figure(figsize=(12,12))

plt.subplot(2,2,1)

boxplot(data_u25['smoker'],data_u25['charges'])

plt.xlabel('people between 18 and 25')

plt.subplot(2,2,2)

boxplot(data_u34['smoker'],data_u34['charges'])

plt.xlabel('people between 26 and 34')

plt.subplot(2,2,3)

boxplot(data_u49['smoker'],data_u49['charges'])

plt.xlabel('people between 35 and 49')

plt.subplot(2,2,4)

boxplot(data_u64['smoker'],data_u64['charges'])

plt.xlabel('people between 50 and 64');
data_1.head()
#no of male and female smokers 



plt.figure(figsize=(12,12))

plt.subplot(2,2,1)

sns.countplot(data_u25.loc[data_u25['smoker']=='yes']['sex'])

plt.xlabel('people under 25')

plt.subplot(2,2,2)

sns.countplot(data_u34.loc[data_u34['smoker']=='yes']['sex'])

plt.xlabel('people under 34')

plt.subplot(2,2,3)

sns.countplot(data_u49.loc[data_u49['smoker']=='yes']['sex'])

plt.xlabel('people under 49')

plt.subplot(2,2,4)

sns.countplot(data_u64.loc[data_u64['smoker']=='yes']['sex'])

plt.xlabel('people under 64');
#medical charges in various regions



boxplot(data_1['region'],data_1['charges'])
data_1.head()
data_1.loc[(data_1['bmi']>=18.5) & (data_1['bmi']<=24.9)].info()
#no of healthy male and females in the dataset



sns.countplot(data_1.loc[(data_1['bmi']>=18.5) & (data_1['bmi']<=24.9)]['sex']);
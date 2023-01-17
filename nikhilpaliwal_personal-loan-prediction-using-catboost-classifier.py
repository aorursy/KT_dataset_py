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

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline

sns.set_style("whitegrid")

plt.style.use("fivethirtyeight")
import os

print(os.listdir("../input/bank-loan-modelling"))
df = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx', 'Data')

df.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]
df.head(4) #to check first 4 rows of data set.
df.info()

listitem=[]

for col in df.columns:

    listitem.append([col,df[col].dtypes,df[col].isna().sum(),round((df[col].isna().sum()/len(df[col]))*100,2),df[col].nunique(),df[col].unique()])

dfdesc=pd.DataFrame(columns=['features','dtype','Null value count','Null value percentage','Unique count','Unique items'],data=listitem)

dfdesc
df.shape #to check no of rows and column
pd.set_option("display.float","{:.2f}".format)

df.describe()
df.PersonalLoan.value_counts()
plt.figure(figsize=(5,5))

df.PersonalLoan.value_counts().plot(kind="bar",color=['salmon','lightblue'])
categorical_val=[]

continuous_val=[]

for column in df.columns:

    print('=================')

    print(f"{column} : {df[column].unique()}")

    if len(df[column].unique()) <= 10:

        categorical_val.append(column)

    else:

        continuous_val.append(column)
categorical_val
continuous_val
plt.figure(figsize=(17,17))

for i , column in enumerate(categorical_val,1):

    plt.subplot(3,3,i)

    df[df["PersonalLoan"]==0][column].hist(bins=35,color='red',label='Have Personal Loan = No')

    df[df["PersonalLoan"]==1][column].hist(bins=35,color='Blue',label="Have Personal Loan = Yes")

    plt.legend()

    plt.xlabel(column)
df[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))
sns.pairplot(data=df)
plt.figure(figsize=(7,7))

plt.scatter(x='Age',y='Experience',data=df)
df["Age"].value_counts().plot.bar(figsize=(20,6))
df.describe()['Experience']
df[df['Experience']<0].count()
# Let us replace all the negative Experience data points by absolute value.

df['Experience']=df['Experience'].apply(abs)
df[df['Experience']<0].count()
Outlier = ['Income', 'CCAvg', 'Mortgage']

Q1=df[Outlier].quantile(0.25)

Q3=df[Outlier].quantile(0.75)

IQR=Q3-Q1

LL,UL = Q1-(IQR*1.5),Q3+(IQR*1.5)



for i in Outlier:

    df[i][df[i]>UL[i]]=UL[i];df[i][df[i]<LL[i]]=LL[i] 
df[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))
corr=df.corr()

fig,ax=plt.subplots(figsize=(12,12))

ax=sns.heatmap(corr,annot=True,square=True,fmt=".2f",cmap="YlGnBu")

categorical_val.remove('PersonalLoan')

print(categorical_val)
dataset = df.copy() # Let us create new dataset
from sklearn.preprocessing import LabelEncoder

encode=LabelEncoder()

label1=encode.fit_transform(df['Family'])

label2=encode.fit_transform(df['Education'])

dataset['Family']=label1

dataset['Education']=label2
dataset
dataset.drop(['ID','Experience','ZIPCode'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler

ssc=StandardScaler()

col_to_scale=['Age','Income','CCAvg','Mortgage']

dataset[col_to_scale] = ssc.fit_transform(dataset[col_to_scale])
X = dataset.drop('PersonalLoan', axis=1)

y = dataset.PersonalLoan
PersonalLoan1=dataset[dataset['PersonalLoan']==1]

PersonalLoan0=dataset[dataset['PersonalLoan']==0]

print(PersonalLoan1.shape,PersonalLoan0.shape)
## RandomOverSampler to handle imbalanced data

from imblearn.over_sampling import RandomOverSampler

os =  RandomOverSampler(random_state=101)

X_ros,y_ros=os.fit_sample(X,y)
from collections import Counter

print('Original dataset shape {}'.format(Counter(y)))

print('Resampled dataset shape {}'.format(Counter(y_ros)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_ros, y_ros, test_size= 0.3, random_state=42)
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()

log_reg.fit(X_ros, y_ros)

y_pred=log_reg.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)
from sklearn.metrics import (accuracy_score , f1_score , precision_score , recall_score)

print("Accuracy:" , accuracy_score(y_test,y_pred))

print("Precision:",precision_score(y_test , y_pred))

print("Recall:",recall_score(y_test,y_pred))

print("F1:", f1_score(y_test,y_pred))
data = pd.read_excel('../input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx', 'Data')

data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]
data.head()
categorical_val=[]

continuous_val=[]

for column in data.columns:

    print('=================')

    print(f"{column} : {data[column].unique()}")

    if len(data[column].unique()) <= 10:

        categorical_val.append(column)

    else:

        continuous_val.append(column)
data[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))
# Capping Method



Outlier = ['Income', 'CCAvg', 'Mortgage']

Q1=data[Outlier].quantile(0.25)

Q3=data[Outlier].quantile(0.75)

IQR=Q3-Q1

LL,UL = Q1-(IQR*1.5),Q3+(IQR*1.5)



for i in Outlier:

    data[i][data[i]>UL[i]]=UL[i];data[i][data[i]<LL[i]]=LL[i] 
data[continuous_val].plot(kind='box',subplots=True, layout=(3,3), fontsize=10, figsize=(14,14))
# Standardizing the variables

from sklearn.preprocessing import StandardScaler

ssc=StandardScaler()

col_to_scale=['Age','Income','CCAvg','Mortgage']

data[col_to_scale] = ssc.fit_transform(data[col_to_scale])
data.drop(['ID','Experience','ZIPCode'],axis=1,inplace=True)

data
cat=['Family','Education','SecuritiesAccount','CDAccount','Online','CreditCard']
target_col='PersonalLoan'

X= df.loc[:,df.columns!=target_col]

y=df.loc[:,target_col]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
features=list(X_train.columns)
# Importing Library

!pip install catboost

from catboost import CatBoostClassifier
model_cb=CatBoostClassifier(iterations=1000, learning_rate=0.01, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42) # use_best_model params will make the model prevent overfitting
model_cb.fit(X_train,y_train,cat_features=cat,eval_set=(X_test,y_test))

y_pred1 =model_cb.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test , y_pred1))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test , y_pred1))
print("Accuracy:",accuracy_score(y_test , y_pred1))

print("Precision:",precision_score(y_test,y_pred1))

print("Recall:",recall_score(y_test,y_pred1))

print("F1-score:",f1_score(y_test,y_pred1))
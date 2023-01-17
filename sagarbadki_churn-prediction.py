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
# Import all the Dependencies

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('fivethirtyeight')



# Feature Engineering 

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



# Model Selection

from sklearn.model_selection import train_test_split



# handle Imbalanced Data

from imblearn.over_sampling import RandomOverSampler



#Models

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier





# Model Evaluation

from sklearn.metrics import classification_report, confusion_matrix



# Hyper-Tuning 

from sklearn.model_selection import RandomizedSearchCV



import warnings

warnings.filterwarnings('ignore')
# reading the data

df=pd.read_csv('../input/customer-chrun/Churn.csv')

df.head()
print('Data Size:',df.shape)
#Checking Null values if any

df.isnull().sum()
df['Churn'].value_counts(normalize=True).plot(kind='bar')
sns.countplot(df['gender'],hue=df['Churn'])
sns.countplot(df['SeniorCitizen'],hue=df['Churn'])
sns.countplot(df['Partner'],hue=df['Churn'])
sns.countplot(df['Dependents'],hue=df['Churn'])
plt.figure(figsize=(10,16))

exp=pd.crosstab(df['tenure'],df['Churn']) 

exp.div(exp.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True,figsize=(15,8))

plt.legend(loc='best')
sns.countplot(df['PhoneService'],hue=df['Churn'])
sns.countplot(df['MultipleLines'],hue=df['Churn'])
sns.countplot(df['InternetService'],hue=df['Churn'])
sns.countplot(df['OnlineSecurity'],hue=df['Churn'])
sns.countplot(df['OnlineBackup'],hue=df['Churn'])
sns.countplot(df['DeviceProtection'],hue=df['Churn'])
sns.countplot(df['TechSupport'],hue=df['Churn'])
sns.countplot(df['StreamingTV'],hue=df['Churn'])
sns.countplot(df['StreamingMovies'],hue=df['Churn'])
sns.countplot(df['Contract'],hue=df['Churn'])
sns.countplot(df['PaperlessBilling'],hue=df['Churn'])
plt.figure(figsize=(10,6))

sns.countplot(df['PaymentMethod'],hue=df['Churn'])
sns.boxplot(y=df['MonthlyCharges'],x=df['Churn'])
# convert string into float and replace with 0 values if any blank



df['TotalCharges']=df['TotalCharges'].replace(' ',0)

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], downcast="float")
sns.boxplot(y=df['TotalCharges'],x=df['Churn'])
# Convert all the categorical features into numerical

# define class

encode=LabelEncoder()



df['gender']=encode.fit_transform(df['gender'])

df['Partner']=encode.fit_transform(df['Partner'])

df['Dependents']=encode.fit_transform(df['Dependents'])

df['PhoneService']=encode.fit_transform(df['PhoneService'])

df['MultipleLines']=encode.fit_transform(df['MultipleLines'])

df['InternetService']=encode.fit_transform(df['InternetService'])

df['OnlineSecurity']=encode.fit_transform(df['OnlineSecurity'])

df['OnlineBackup']=encode.fit_transform(df['OnlineBackup'])

df['DeviceProtection']=encode.fit_transform(df['DeviceProtection'])

df['TechSupport']=encode.fit_transform(df['TechSupport'])

df['StreamingTV']=encode.fit_transform(df['StreamingTV'])

df['StreamingMovies']=encode.fit_transform(df['StreamingMovies'])

df['Contract']=encode.fit_transform(df['Contract'])

df['PaperlessBilling']=encode.fit_transform(df['PaperlessBilling'])

df['PaymentMethod']=encode.fit_transform(df['PaymentMethod'])

df['Churn']=encode.fit_transform(df['Churn'])





# Convert tenure feature into 3 category (i have taken 2 year difference according to the previous Analysis.)

# 0-24Months-->1, 25-48 Months--->2 and else is 3



df['tenure']=df['tenure'].map(lambda x: 1 if x<=24 else 2 if x<=48 else 3)
#define class

scale=StandardScaler()



df['MonthlyCharges']=scale.fit_transform(df['MonthlyCharges'].values.reshape(-1,1))

df['TotalCharges']=scale.fit_transform(df['TotalCharges'].values.reshape(-1,1))
# df.groupby('Churn')['TotalCharges'].mean()

mean=1531.796143

df1=df[(df['Churn']=='Yes') & (df['TotalCharges']>=mean)]['TotalCharges'].map(lambda x: mean if x>mean else x)

df['TotalCharges'].update(df1)
plt.figure(figsize=(15,15))

sns.heatmap(df.corr(),annot=True)
# split the features

X=df.drop(['customerID','Churn','tenure'],axis=1)

y=df['Churn']
# We have Imbalanced Data and we have to do sampling to avoid this problem

# we have two method so for 1] Under Sampling, 2] Oversampling

# I will go for Oversampling.



sample=RandomOverSampler()

X_sample,y_sample=sample.fit_sample(X,y)


X_train,X_test,y_train,y_test=train_test_split(X_sample,y_sample,test_size=0.2)
# start with basic model (Logistic Regression)



reg=LogisticRegression(max_iter=1000)

reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)



report=classification_report(y_test,y_pred)

matrix=confusion_matrix(y_test,y_pred)



print('Classification Report:\n',report)

print('Confusion Matrix :\n',matrix)



# Now lets go for my favourite one 



random=RandomForestClassifier()

random.fit(X_train,y_train)

y_pred=random.predict(X_test)



report=classification_report(y_test,y_pred)

matrix=confusion_matrix(y_test,y_pred)



print('Classification Report:\n',report)

print('Confusion Matrix :\n',matrix)



params={'n_estimators':[i for i in  range(100,2000,200)],

      'max_depth':[1,2,4,5,10,15,20,30,35,40],

       'min_samples_split':[1,2,4,5,10,15,20],

       'min_samples_leaf':[1,2,6,8,10,15,20,25,30]}



clf= RandomForestClassifier()



model=RandomizedSearchCV(clf,param_distributions=params,cv=3)



model.fit(X_train,y_train)
model.best_score_
features=pd.DataFrame({'Important_features':random.feature_importances_},index=X.columns)
features.sort_values(by='Important_features',ascending=True).plot(kind='barh')
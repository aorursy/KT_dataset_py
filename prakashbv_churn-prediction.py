# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Load the dataset

df = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.shape
df.head(10)
df = df.drop(['customerID'],axis=1)
df = df.dropna(how="all") # remove samples with all missing values

df = df[~df.duplicated()] # remove duplicates

total_charges_filter = df.TotalCharges == " "

df = df[~total_charges_filter]

df.TotalCharges = pd.to_numeric(df.TotalCharges)
df.dtypes
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
df.dtypes
df['SeniorCitizen'] = df['SeniorCitizen'].replace(0,'Yes')

df['SeniorCitizen'] = df['SeniorCitizen'].replace(1,'No')

df['SeniorCitizen'] = df['SeniorCitizen'].astype('object')
df['SeniorCitizen'].head()
df.dtypes
df.head(10)
#categorical_var = list(df.columns)

categorical_vars = ['gender','SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',

                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod','Churn']

numerical_var = ['tenure','MonthlyCharges','TotalCharges']
df.dtypes
df[numerical_var].describe()
df[numerical_var].hist(bins=30, figsize=(10, 7))
df['Churn'].head(10)
fig, ax = plt.subplots(1, 3, figsize=(14, 4))

df[df.Churn == "No"][numerical_var].hist(bins=30, color='blue', alpha=0.5, ax=ax)

df[df.Churn == "Yes"][numerical_var].hist(bins=30, color='red', alpha=0.5, ax=ax)
ROWS, COLS = 4, 4

fig, ax = plt.subplots(ROWS, COLS, figsize=(18, 18))

row, col = 0, 0

for i, categorical_vars in enumerate(categorical_vars):

    if col == COLS - 1:

        row += 1

    col = i % COLS

    df[categorical_vars].value_counts().plot('bar', ax=ax[row, col]).set_title(categorical_vars)
df['Churn'].value_counts().plot('bar').set_title('churned')
lis = []

for i in range(0, df.shape[1]):

    #print(i)

    if(df.iloc[:,i].dtypes == 'object'):

        df.iloc[:,i] = pd.Categorical(df.iloc[:,i])

        #print(marketing_train[[i]])

        df.iloc[:,i] = df.iloc[:,i].cat.codes 

        df.iloc[:,i] = df.iloc[:,i].astype('object')

        

        lis.append(df.columns[i])
df.dtypes
df['gender'] = df['gender'].astype('category')

df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

df['Partner'] = df['Partner'].astype('category')

df['Dependents'] = df['Dependents'].astype('category')

df['PhoneService'] = df['PhoneService'].astype('category')

df['MultipleLines'] = df['MultipleLines'].astype('category')

df['InternetService'] = df['InternetService'].astype('category')

df['OnlineSecurity'] = df['OnlineSecurity'].astype('category')

df['OnlineBackup'] = df['OnlineBackup'].astype('category')

df['DeviceProtection'] = df['DeviceProtection'].astype('category')

df['TechSupport'] = df['TechSupport'].astype('category')

df['StreamingTV'] = df['StreamingTV'].astype('category')

df['StreamingMovies'] = df['StreamingMovies'].astype('category')

df['Contract'] = df['Contract'].astype('category')

df['PaperlessBilling'] = df['PaperlessBilling'].astype('category')

df['PaymentMethod'] = df['PaymentMethod'].astype('category')

df['Churn'] = df['Churn'].astype('category')
df.dtypes
#for i in categorical_vars:

#    df[i] = df[i].astype('category')
df.shape
#Splitting the dataset into train and test for model building

from sklearn.model_selection import train_test_split



x = df.iloc[:,:-1]

y = df.iloc[:,19]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
x.dtypes
#Training and fitting the model

from sklearn.svm import SVC
#Building model on train data

model = SVC(kernel='rbf', gamma=0.7, C=1.0)

#model =  SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,

#    decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,

#    probability=False, random_state=None, shrinking=True, tol=0.001,

#    verbose=False)

model.fit(x_train, y_train)
#Predicting for the test data

predictions = model.predict(x_test)
#importing libraries for model evaluation

from sklearn.metrics import classification_report, accuracy_score
#Confusion matrix

print(pd.crosstab(y_test, predictions))



print(classification_report(y_test, predictions))
model2 = SVC(kernel='linear', C=1.0).fit(x_train, y_train)

predictions2 = model2.predict(x_test)

print(pd.crosstab(y_test, predictions2))

print(classification_report(y_test, predictions2))

from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(x_train,y_train)

pred = clf.predict(x_test)

print(pd.crosstab(y_test, pred))

print(classification_report(y_test, pred))
df.dtypes
from xgboost import XGBClassifier

clf = XGBClassifier()

clf.fit(x_train, y_train)

pred = clf.predict(x_test)

print(pd.crosstab(y_test, pred))

print(classification_report(y_test, pred))
from sklearn.neighbors import KNeighborsClassifier

# Instantiate learning model (k = 3)

classifier = KNeighborsClassifier(n_neighbors=3)



# Fitting the model

classifier.fit(x_train, y_train)



pred = classifier.predict(x_test)

print(pd.crosstab(y_test, pred))

print(classification_report(y_test, pred))
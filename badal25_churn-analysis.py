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
df = pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.shape
# for view all columns and rows



pd.set_option('display.max_columns',None)

pd.set_option('display.max_rows', None)
df.head()
df.isnull().sum().sum()
df.columns
df.Churn.value_counts()
# check numerical variable

df.select_dtypes(include=['int64','float64']).columns
columns = df.columns

binary_cols = []



for col in columns:

    if df[col].value_counts().shape[0]==2:

        binary_cols.append(col)
#categorical features with two classes

binary_cols
# Categorical features with multiple classes

multiple_cols_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',

 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)



sns.countplot("gender", data=df, ax=axes[0,0])

sns.countplot("SeniorCitizen", data=df, ax=axes[0,1])

sns.countplot("Partner", data=df, ax=axes[0,2])

sns.countplot("Dependents", data=df, ax=axes[1,0])

sns.countplot("PhoneService", data=df, ax=axes[1,1])

sns.countplot("PaperlessBilling", data=df, ax=axes[1,2])
churn_numeric = []

for i in range(len(df)):

    if df['Churn'][i] == 'Yes':

        churn_numeric.append(1)

    else:

        churn_numeric.append(0)
churn_numeric[:5]
df['Churn']= churn_numeric
df[['gender','Churn']].groupby(['gender']).mean()
df[['Partner','Churn']].groupby(['Partner']).mean()
df[['Dependents','Churn']].groupby(['Dependents']).mean()
df[['PhoneService','Churn']].groupby(['PhoneService']).mean()
df[['PaperlessBilling','Churn']].groupby(['PaperlessBilling']).mean()
table = pd.pivot_table(df, values='Churn', index=['gender'],

                    columns=['SeniorCitizen'], aggfunc=np.mean)

table
table = pd.pivot_table(df, values='Churn', index=['Partner'],

                    columns=['Dependents'], aggfunc=np.mean)

table
sns.countplot("InternetService", data=df)
df[['InternetService','Churn']].groupby('InternetService').mean()
df[['InternetService','MonthlyCharges']].groupby('InternetService').mean()
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)



sns.countplot("StreamingTV", data=df, ax=axes[0,0])

sns.countplot("StreamingMovies", data=df, ax=axes[0,1])

sns.countplot("OnlineSecurity", data=df, ax=axes[0,2])

sns.countplot("OnlineBackup", data=df, ax=axes[1,0])

sns.countplot("DeviceProtection", data=df, ax=axes[1,1])

sns.countplot("TechSupport", data=df, ax=axes[1,2])
df[['StreamingTV','Churn']].groupby('StreamingTV').mean()
df[['StreamingMovies','Churn']].groupby('StreamingMovies').mean()
df[['OnlineSecurity','Churn']].groupby('OnlineSecurity').mean()
df[['OnlineBackup','Churn']].groupby('OnlineBackup').mean()
df[['DeviceProtection','Churn']].groupby('DeviceProtection').mean()
df.PhoneService.value_counts()
df.MultipleLines.value_counts()
df[['MultipleLines','Churn']].groupby('MultipleLines').mean()
plt.figure(figsize=(10,6))

sns.countplot("Contract", data=df)
df[['Contract','Churn']].groupby('Contract').mean()
plt.figure(figsize=(10,6))

sns.countplot("PaymentMethod", data=df)
df[['PaymentMethod','Churn']].groupby('PaymentMethod').mean()
fig, axes = plt.subplots(1,2, figsize=(12, 7))



sns.distplot(df["tenure"], ax=axes[0])

sns.distplot(df["MonthlyCharges"], ax=axes[1])
df[['tenure','MonthlyCharges','Churn']].groupby('Churn').mean()
df[['Contract','tenure']].groupby('Contract').mean()
df.drop(['customerID','gender','PhoneService','Contract','TotalCharges'], axis=1, inplace=True)
df.head()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
cat_features = ['SeniorCitizen', 'Partner', 'Dependents',

        'MultipleLines', 'InternetService', 'OnlineSecurity',

       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',

       'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']

X = pd.get_dummies(df, columns=cat_features, drop_first=True)
sc = MinMaxScaler()

a = sc.fit_transform(df[['tenure']])

b = sc.fit_transform(df[['MonthlyCharges']])
X['tenure'] = a

X['MonthlyCharges'] = b
X.shape
sns.countplot('Churn', data=df).set_title('Class Distribution Before Resampling')
X_no = X[X.Churn == 0]

X_yes = X[X.Churn == 1]
print(len(X_no),len(X_yes))
X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)

print(len(X_yes_upsampled))
X_upsampled = X_no.append(X_yes_upsampled).reset_index(drop=True)
sns.countplot('Churn', data=X_upsampled).set_title('Class Distribution After Resampling')
from sklearn.model_selection import train_test_split
X = X_upsampled.drop(['Churn'], axis=1) #features (independent variables)

y = X_upsampled['Churn'] #target (dependent variable)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
from sklearn.linear_model import RidgeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
clf_ridge = RidgeClassifier() #create a ridge classifier object

clf_ridge.fit(X_train, y_train) #train the model
pred = clf_ridge.predict(X_train)  #make predictions on training set
accuracy_score(y_train, pred) #accuracy on training set
confusion_matrix(y_train, pred)
pred_test = clf_ridge.predict(X_test)
accuracy_score(y_test, pred_test)
from sklearn.ensemble import RandomForestClassifier
clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10)
clf_forest.fit(X_train, y_train)
pred = clf_forest.predict(X_train)
accuracy_score(y_train, pred)
confusion_matrix(y_train, pred)
pred_test = clf_forest.predict(X_test)
accuracy_score(y_test, pred_test)
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[150,200,250,300], 'max_depth':[15,20,25]}

forest = RandomForestClassifier()

clf = GridSearchCV(estimator=forest, param_grid=parameters, n_jobs=-1, cv=5)

clf.fit(X, y)
clf.best_params_
clf.best_score_
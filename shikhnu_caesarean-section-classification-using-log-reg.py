import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

import pickle
df = pd.read_csv('../input/caesarean-section-classification/Caesarian Section Classification Dataset(CSV).csv')

#df = pd.read_csv('Cesarean .csv')

df.head()
# To know number od rows and column

df.shape
# To know if missing value is preset and also to know dtypes

df.info()
# Five point summary

df.describe().T
#To find no. of unique values in categorical columns



for col in df.select_dtypes(include=object).columns:

    print('No. of unique values in column '+col+':')

    print(df[col].value_counts(),'\n')
# Replacing 'low' to 'Low' and 'yes' to 'Yes' 



df['Blood of Pressure'] = df['Blood of Pressure'].replace('low','Low')

df['Caesarian'] = df['Caesarian'].replace('yes','Yes')
# Plotting countplot



fig = plt.figure(figsize=(15,10))



fig.add_subplot(221)

plt.title('Delivery No.', fontsize=12)

sns.countplot(df['Delivery No'])



fig.add_subplot(2,2,2)

plt.title('Blood of Pressure', fontsize=12)

sns.countplot(df['Blood of Pressure'])



fig.add_subplot(223)

plt.title('Heart Problem', fontsize=12)

sns.countplot(df['Heart Problem'])



fig.add_subplot(2,2,4)

plt.title('Caesarian', fontsize=12)

sns.countplot(df['Caesarian'])



plt.show()
# Encoding using One-Hot encoding



df_dummy = pd.get_dummies(df,drop_first=True)

df_dummy.head()
#Separting df_train in independent and dependent variable

X=df_dummy.drop(['Caesarian_Yes'],axis=1)

y=df_dummy['Caesarian_Yes']
#Splitting df_train in train and test

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



logreg = LogisticRegression(solver='liblinear', fit_intercept=True) 



logreg.fit(X_train, y_train)



y_prob_train = logreg.predict_proba(X_train)[:,1]

y_pred_train = logreg.predict (X_train)



print('Confusion Matrix - Train: ', '\n', confusion_matrix(y_train, y_pred_train))

print('Overall accuracy - Train: ', accuracy_score(y_train, y_pred_train))





y_prob = logreg.predict_proba(X_test)[:,1]

y_pred = logreg.predict (X_test)



print('Confusion Matrix - Test: ','\n', confusion_matrix(y_test, y_pred))

print('Overall accuracy - Test: ', accuracy_score(y_test, y_pred))
#Fitting whole dataset

log_r = logreg.fit(X, y)
pickle.dump(log_r, open('caesarean.pkl','wb')) #wb->write binary
# import libraries 

import pandas as pd

import numpy as np
df=pd.read_csv('../input/ChurnData.csv')
df.head(5)
df.info()
# check to see if there are any missing data

missing_data=df.isnull()

missing_data.head(5)
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print('')
df.shape
df.corr()
# selecting the features that we can use for our model development

df = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]



# churn type is float. We need to change this to integer for our Logictic Regression

df['churn'] = df['churn'].astype('int')

df.head(5)

#lets see our column and row number now

df.shape
# preprocessing and definning the X and y (Features and target)

X=np.asarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])

X[0:5]
y=np.array(df['churn'])

y[0:5]
# normalize and preprocess the dataset

from sklearn import preprocessing

X = preprocessing.StandardScaler().fit(X).transform(X)

X[0:5]
# split the data set for train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
# import required libraries and create the model with Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR
# predict using the test set

yhat = LR.predict(X_test)

yhat
# predict probability (estimates) for all classes 

# first column will be probability of class 1 and second column is probability of class 0

yhat_prob = LR.predict_proba(X_test)

yhat_prob
from sklearn.metrics import jaccard_similarity_score

jaccard_similarity_score(y_test, yhat)
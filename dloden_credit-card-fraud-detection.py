# Load packages

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import classification_report

from sklearn.svm import OneClassSVM

from sklearn.pipeline import Pipeline



# Load dataset

data = pd.read_csv('../input/creditcard.csv')

data.drop('Time', axis=1, inplace=True)



# Split into features and target

X = data.ix[:, 0:29]

y = data.ix[:, 29]



# Split into training and testing sets (50%, 50%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 2008)



# Remove fradulent transactions from training data for One-Class SVMs

X_train_good = X_train[y_train == 0]

y_train_good = y_train[y_train == 0]



# Instantiate normalisation 

nrm = MinMaxScaler()
print('Proportion of fraudulent transactions', round(np.mean(y_train), 3))
svm = OneClassSVM(random_state=2008, nu=0.2) # Nu set by trial and error

svm_pl = Pipeline([('Normalise', nrm),

                   ('SVM', svm)])

svm_pl.fit(X_train_good)
# Predict fradulent transactions in test set

y_test_pred = svm_pl.predict(X_test) # Outputs data in {-1, 1}

y_test_pred = ((y_test_pred * -1) + 1) / 2 # Convert to {1, 0}



# Evaluate

print(classification_report(y_test, y_test_pred))
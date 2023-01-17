# Importing the required libraries

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#reading csv file and creating a dataframe

df = pd.read_csv("../input/Chickpea.data.csv")

df.head()
#Na Handling

df.isnull().values.any()
df.shape
df.isnull()
df1 = df.dropna()
Y=df1['Predictor']

Y.describe()

Y=pd.DataFrame(Y)
#collecting indepedent variables in X

X = df1.iloc[:,1:332]

X_col = X.columns

X.head()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

Y_encoded = Y.apply(le.fit_transform)

Y_encoded.head()
#Savitzky-Golay filter with second degree derivative.

from scipy.signal import savgol_filter 



sg=savgol_filter(X,window_length=11, polyorder=3, deriv=2, delta=1.0)
sg_x=pd.DataFrame(sg, columns=X_col)

sg_x.head()
# Splitting the data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =train_test_split(sg_x, Y_encoded,

                                                    test_size=0.2,

                                                    random_state=123,stratify=Y)
from sklearn.tree import DecisionTreeClassifier



Dtree = DecisionTreeClassifier(random_state=52,criterion='entropy',min_samples_split=50)

Dtree_fit=Dtree.fit(X_train, y_train)
y_pred = Dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Test Result:\n")        

print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Classification Report: \n {}\n".format(classification_report(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test,y_pred)))
#Reduction of variables using Recursive Feature Elimination(RFE) techineque.



from sklearn.feature_selection import RFE



# RFE with 10 features



rfe_10 = RFE(Dtree ,10)



rfe_10.fit(X_train, y_train)



# selected features

features_bool = np.array(rfe_10.support_)

features = np.array(X_col)

result = features[features_bool]

print('10 selected Features:')

print(result)
y_pred = rfe_10.predict(X_test)



print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred)))
# RFE with 15 features



rfe_15 = RFE(Dtree ,15)



# fit with 15 features

RFE15_fit=rfe_15.fit(X_train, y_train)



# selected features

features_bool = np.array(rfe_15.support_)

features = np.array(X_col)

result = features[features_bool]

print('15 selected Features:')

print(result)
y_pred = rfe_15.predict(X_test)



print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred)))
from sklearn.ensemble import RandomForestClassifier



Rf = RandomForestClassifier(random_state=52)

Rf_fit=Rf.fit(X_train, y_train)
y_pred = Rf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Test Result:\n")        

print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Classification Report: \n {}\n".format(classification_report(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test,y_pred)))
# RFE with 20 features



rfe_20 = RFE(Rf,20)



# fit with 20 features

rfe_20.fit(X_train, y_train)
y_pred = rfe_20.predict(X_test)



print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, y_pred)))

print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, y_pred)))
#link file on github
#  https://github.com/AlhassanMohamed/deeplearning_work/tree/master/kaggle/titanic%20survival%20predictions
#importing library
import pandas as pd 
import numpy as np
#read data sets from csv files
train = pd.read_csv('../input/train.csv')
test  = pd.read_csv('../input/test.csv')

#print the datasets to know what i should take from files
print('-'*20,'train','-'*20)
#train
# determine features and label from train dataframe
# label y
y_train = train.iloc[:,[1]].values
# feature X
X_train = train.iloc[:,[2,4,5,6,7,9,11]]

#print y_train
#y_train
#print X_train 
#X_train
# start to clean the X_train from NaN using 'mean'
from sklearn.preprocessing import Imputer
X_age_imputer = Imputer()
X_age_imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
X_train.iloc[:,[2]] = X_age_imputer.fit_transform(X_train.iloc[:,[2]])

#print X_train after takeing care of missing data
#X_train
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_labelencoder_1 = LabelEncoder()
X_train.iloc[:,[1]] = X_labelencoder_1.fit_transform(X_train.iloc[:,[1]])

#X_train
X_train = X_train.iloc[:,:-1]
#X_train
#print the datasets to know what i should take from files
print('-'*40,'test','-'*40)
#test
#importing y_train form csv file
gender_submission  = pd.read_csv('../input/gender_submission.csv')

# determine features and label from train dataframe
# label y
y_test = gender_submission.iloc[:,[1]]
# feature X
X_test = test.iloc[:,[1,3,4,5,6,8]]

#print X_test
#X_test
# start to clean the X_test from NaN using 'mean'
from sklearn.preprocessing import Imputer
X_age_imputer = Imputer()
X_age_imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
X_test.iloc[:,[2,-1]] = X_age_imputer.fit_transform(X_test.iloc[:,[2,-1]])

#print X_test after takeing care of missing data
#X_test
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_labelencoder_1 = LabelEncoder()
X_test.iloc[:,[1]] = X_labelencoder_1.fit_transform(X_test.iloc[:,[1]])

#X_test
#fill missing value 0
X_test.isnull().sum()
# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test.values)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
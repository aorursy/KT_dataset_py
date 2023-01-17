# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")
train_data.head()
print("train_data shape: ", train_data.shape)
target = train_data["Category"].unique()
print(target.shape)
target
#df.drop(df.columns[[0, 1, 3]], axis=1)
X = train_data.drop(train_data.columns[[1, 2, 5, 6]], axis = 1)
X.head()
y = train_data.iloc[:, 1]
y.head()
def preprocess_data(dataset):
    dataset['Dates'] = pd.to_datetime(dataset['Dates'])
    dataset['Month'] = dataset.Dates.apply(lambda x: x.month)
    dataset['Day'] = dataset.Dates.apply(lambda x: x.day)
    dataset['Hour'] = dataset.Dates.apply(lambda x: x.hour)
    dataset['Minute'] = dataset.Dates.apply(lambda x: x.minute)
    dataset = dataset.drop('Dates', 1)
    
    dataset = pd.get_dummies(data=dataset, columns=['DayOfWeek', 'PdDistrict'])
    return dataset
X = preprocess_data(X)
X.head()

from sklearn.preprocessing import LabelEncoder
y = y.to_frame()
le = LabelEncoder()
y["Category"] = le.fit_transform(y["Category"])
print(y.head())

keys = le.classes_
values = le.transform(le.classes_)
dictionary = dict(zip(keys, values))
print(dictionary)

X.head()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

print(type(X_train), type(y_train))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth = 10, n_estimators = 256)
rf.fit(X_train.values, y_train.values.ravel())

y_pred = rf.predict(X_test)
y_pred.shape
y_test.shape

from sklearn.metrics import accuracy_score
print ("Train Accuracy: ", accuracy_score(y_train, rf.predict(X_train)))
print ("Test Accuracy: ", accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns  
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm)
sns.heatmap(cm, cmap="YlGnBu", square=True)

print(classification_report(y_test, y_pred))  
test_data = pd.read_csv('../input/test.csv')
test_data.head()
test_data = preprocess_data(test_data)
test_data.head()
test_data = test_data.drop('Id', 1)
test_data = test_data.drop('Address', 1)
test_data.head()
y_pred_proba = rf.predict_proba(test_data)
colmn = ["ARSON","ASSAULT","BAD CHECKS","BRIBERY","BURGLARY","DISORDERLY CONDUCT","DRIVING UNDER THE INFLUENCE","DRUG/NARCOTIC","DRUNKENNESS","EMBEZZLEMENT","EXTORTION","FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD","GAMBLING","KIDNAPPING","LARCENY/THEFT","LIQUOR LAWS","LOITERING","MISSING PERSON","NON-CRIMINAL","OTHER OFFENSES","PORNOGRAPHY/OBSCENE MAT","PROSTITUTION","RECOVERED VEHICLE","ROBBERY","RUNAWAY","SECONDARY CODES","SEX OFFENSES FORCIBLE","SEX OFFENSES NON FORCIBLE","STOLEN PROPERTY","SUICIDE","SUSPICIOUS OCC","TREA","TRESPASS","VANDALISM","VEHICLE THEFT","WARRANTS","WEAPON LAWS"]
result = pd.DataFrame(y_pred_proba, columns=colmn)

result.to_csv(path_or_buf="rf_predict.csv",index=True, index_label = 'Id')
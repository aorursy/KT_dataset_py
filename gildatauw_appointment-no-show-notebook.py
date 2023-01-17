# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#Read data into data frame
df = pd.read_csv('../input/KaggleV2-May-2016.csv')

#Remove records where age is negative
#df[df.Age > 0]

#Drop non-relavent columns
df.drop(['AppointmentID', 'ScheduledDay'], axis = 1, inplace = True)

#Fix column spelling mistakes
df.rename(index=str, columns={"Neighbourhood" : "Neighborhood", "Hipertension" : "Hypertension", "Handcap" : "Handicap", 
                              "SMS_received" : "SMSReceived", "No-show" : "NoShow"}, inplace = True)

#Change text values to binary values
df.replace({'NoShow':{'No' : 0, 'Yes' : 1}}, inplace = True)
df.replace({'Gender':{'M': 0, 'F':1}}, inplace = True)

#Add DayOfWeek column to dataset
df['DayOfWeek'] = pd.to_datetime(df['AppointmentDay']).dt.dayofweek
df['NeighborhoodCode'] = df.Neighborhood.astype('category').cat.codes
df
import matplotlib.pyplot as plt
import seaborn as sea
sea.countplot(x = 'Gender', hue = 'NoShow', data = df)
plt.show()

sea.countplot(x = 'Scholarship', hue = 'NoShow', data = df)
plt.show()

sea.countplot(x = 'Hypertension', hue = 'NoShow', data = df)
plt.show()

sea.countplot(x = 'Diabetes', hue = 'NoShow', data = df)
plt.show()

sea.countplot(x = 'Alcoholism', hue = 'NoShow', data = df)
plt.show()

sea.countplot(x = 'Handicap', hue = 'NoShow', data = df)
plt.show()

sea.countplot(x = 'SMSReceived', hue = 'NoShow', data = df)
plt.show()

sea.countplot(x = 'DayOfWeek', hue = 'NoShow', data = df)
plt.show()


df.head()
df.info()
# Convert PatientId from Float to Integer
df['PatientId'] = df['PatientId'].astype('int64')
df.head()
del df['AppointmentDay']
del df['Neighborhood']
# Print Unique Values
print("Unique Values in `Age` => {}".format(df.Gender.unique()))
print("Unique Values in `Gender` => {}".format(df.Gender.unique()))
print("Unique Values in `Scholarship` => {}".format(df.Scholarship.unique()))
print("Unique Values in `Hypertension` => {}".format(df.Hypertension.unique()))
print("Unique Values in `Diabetes` => {}".format(df.Diabetes.unique()))
print("Unique Values in `Alcoholism` => {}".format(df.Alcoholism.unique()))
print("Unique Values in `Handicap` => {}".format(df.Handicap.unique()))
print("Unique Values in `SMSReceived` => {}".format(df.SMSReceived.unique()))
print("Unique Values in `NoShow` => {}".format(df.NoShow.unique()))
#setup testing and training data

from sklearn.model_selection import train_test_split

xValues = df.drop(['NoShow', 'SMSReceived', 'NeighborhoodCode', 'PatientId'], axis = 1)
yValues = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size = 0.7, random_state = 40)


#using naive bayes for model 
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()



print("Naive Bayes Classifier:")
nb.fit(X_train, y_train)
print("Accuracy: %8.2f" % (nb.score(X_test, y_test)))
print(classification_report(y_test, nb.predict(X_test)))





#DecisionTree
from sklearn.cross_validation import  cross_val_score
from sklearn.tree import DecisionTreeClassifier

CV = 10
print("{}-fold cross-validation ".format(CV))
dt_clf = DecisionTreeClassifier (random_state=42, max_depth=7)
dt_clf.fit(X_train, y_train)

scores = cross_val_score(dt_clf, X_train, y_train, cv=CV)

print("CV mean: {:.5f} (std: {:.5f})".format(scores.mean(), scores.std()), end="\n\n" )
print(classification_report(y_test, dt_clf.predict(X_test)))


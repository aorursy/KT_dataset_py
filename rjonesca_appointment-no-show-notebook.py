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
del df['AppointmentID']
del df['ScheduledDay']

#Fix column spelling mistakes
df.rename(index=str, columns={"Neighbourhood" : "Neighborhood", "Hipertension" : "Hypertension", "Handcap" : "Handicap", 
                              "SMS_received" : "SMSReceived", "No-show" : "NoShow"}, inplace = True)

# Convert PatientId from Float to Integer
df['PatientId'] = df['PatientId'].astype('int64')

#Change text values to binary values
df.replace({'NoShow':{'No' : 0, 'Yes' : 1}}, inplace = True)
df.replace({'Gender':{'M': 0, 'F':1}}, inplace = True)

#Add DayOfWeek column to dataset
df['DayOfWeek'] = pd.to_datetime(df['AppointmentDay']).dt.dayofweek
df['NeighborhoodCode'] = df.Neighborhood.astype('category').cat.codes
df
import matplotlib.pyplot as plt
import seaborn as sea
#Show Show/No-Show pie chart
labels = 'Show', 'No Show'
colors = ['lightskyblue', 'lightcoral']
explode = (0, 0.1) # only "explode" the 2nd slice (i.e. 'Hogs')

#Calculate Show/No-Show percentage
ns = df.groupby(["NoShow"]).size()[0]
total = df.groupby(["NoShow"]).size()[0] + df.groupby(["NoShow"]).size()[1]
nsPercent = np.round((ns / total),2) * 100
x = [(100 - nsPercent), nsPercent]

plt.pie(x, explode = explode, labels = labels, colors = colors,
        autopct='%1.1f%%', shadow = True, startangle = 90)

# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.show()

#sea.countplot(x = 'Gender', hue = 'NoShow', data = df)
#plt.show()

#sea.countplot(x = 'Scholarship', hue = 'NoShow', data = df)
#plt.show()

#sea.countplot(x = 'Hypertension', hue = 'NoShow', data = df)
#plt.show()

#sea.countplot(x = 'Diabetes', hue = 'NoShow', data = df)
#plt.show()

#sea.countplot(x = 'Alcoholism', hue = 'NoShow', data = df)
#plt.show()

#sea.countplot(x = 'Handicap', hue = 'NoShow', data = df)
#plt.show()

#sea.countplot(x = 'SMSReceived', hue = 'NoShow', data = df)
#plt.show()

#sea.countplot(x = 'DayOfWeek', hue = 'NoShow', data = df)
#plt.show()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np # linear algebra
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2



rfc = RandomForestClassifier(n_estimators=200)
dtc = DecisionTreeClassifier()
nb = MultinomialNB()
lr = LogisticRegression()

#setup testing and training data
xValues = df.drop(['NoShow', 'AppointmentDay', 'Neighborhood', 'PatientId'], axis = 1)
yValues = df['NoShow']

X_train, X_test, y_train, y_test = train_test_split(xValues, yValues, test_size = 0.7, random_state = 40)

#Get feature names
featureNames = list(X_train)

#Determine best features to use
chi2 = SelectKBest(chi2, k=3)
X_train = chi2.fit_transform(X_train, y_train)
X_test = chi2.transform(X_test)

mask = chi2.get_support() #list of booleans
newFeatures = [] # The list of your K best features
for bool, feature in zip(mask, featureNames):
    if bool:
        newFeatures.append(feature)
        
print(newFeatures) 

print("Logistic Regression:")
lr.fit(X_train, y_train)
print("Accuracy: %8.2f" % (lr.score(X_test, y_test)))
print(classification_report(y_test, lr.predict(X_test)))

print("Random Forest Classifier:")
rfc.fit(X_train, y_train)
print("Accuracy: %8.2f" % (rfc.score(X_test, y_test)))
print(classification_report(y_test, rfc.predict(X_test)))

print("Naive Bayes Classifier:")
nb.fit(X_train, y_train)
print("Accuracy: %8.2f" % (nb.score(X_test, y_test)))
print(classification_report(y_test, nb.predict(X_test)))

print("Decision Tree Classifier:")
dtc.fit(X_train, y_train)
print("Accuracy: %8.2f" % (dtc.score(X_test, y_test)))
print(classification_report(y_test, dtc.predict(X_test)))

#accuracy = cross_val_score(nb, X_test, y_test, cv = 10)
#precision = cross_val_score(nb, X_test, y_test, cv = 10, scoring = 'precision')
#recall = cross_val_score(nb, X_test, y_test, cv = 10, scoring = 'recall')





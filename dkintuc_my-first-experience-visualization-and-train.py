# Module import

import numpy as np

import pandas as pd

import seaborn as sns

import os

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/heart.csv')

data.head(5)
data.info()
data.isnull().sum()
plot = data[data.target == 1].age.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 15)

plot.set_title("Age distribution", fontsize = 20)
male = len(data[data.sex == 1])

female = len(data[data.sex == 0])

plt.pie(x=[male, female], explode=(0, 0), labels=['Male', 'Female'], autopct='%1.2f%%', shadow=True, startangle=90)

plt.show()
x = [len(data[data['cp'] == 0]),len(data[data['cp'] == 1]), len(data[data['cp'] == 2]), len(data[data['cp'] == 3])]

plt.pie(x, data=data, labels=['CP(1) typical angina', 'CP(2) atypical angina', 'CP(3) non-anginal pain', 'CP(4) asymptomatic'], autopct='%1.2f%%', shadow=True,startangle=90)

plt.show()
plot = data[data.target == 1].trestbps.value_counts().sort_index().plot(kind = "bar", figsize=(15,4), fontsize = 15)

plot.set_title("Resting blood pressure", fontsize = 20)
plt.hist([data.chol[data.target==0], data.chol[data.target==1]], bins=20,color=['blue', 'red'], stacked=True)

plt.legend(["Haven't Disease", "Have Disease"])

plt.title('Heart Disease Frequency for cholestoral ')

plt.ylabel('Frequency')

plt.xlabel('Chol in mg/dl')

plt.plot()
sizes = [len(data[data.fbs == 0]), len(data[data.fbs==1])]

labels = ['No', 'Yes']

plt.pie(x=sizes, labels=labels, explode=(0.1, 0), autopct="%1.2f%%", startangle=90,shadow=True)

plt.show()
sizes = [len(data[data.restecg == 0]), len(data[data.restecg==1]), len(data[data.restecg==2])]

labels = ['Normal', 'ST-T wave abnormality', 'definite left ventricular hypertrophy by Estes criteria']

plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)

plt.show()
plt.hist([data.thalach[data.target==0], data.thalach[data.target==1]], bins=20,color=['blue', 'red'], stacked=True)

plt.legend(["Haven't Disease", "Have Disease"])

plt.title('Heart Disease Frequency for maximum heart rate achieved')

plt.ylabel('Frequency')

plt.xlabel('Heart rate')

plt.plot()
sizes = [len(data[data.exang == 0]), len(data[data.exang==1])]

labels = ['No', 'Yes']

plt.pie(x=sizes, labels=labels, explode=(0.1, 0), autopct="%1.2f%%", startangle=90,shadow=True)

plt.show()
sizes = [len(data[data.slope == 0]), len(data[data.slope==1]), len(data[data.slope==2])]

labels = ['Upsloping', 'Flat', 'Downssloping']

plt.pie(x=sizes, labels=labels, explode=(0, 0, 0), autopct="%1.2f%%", startangle=90,shadow=True)

plt.show()
sns.countplot('thal', data=data)

plt.title('Frequency for thal')

plt.ylabel('Frequency')

plt.show()
cp = pd.get_dummies(data['cp'], prefix = "cp", drop_first=True)

thal = pd.get_dummies(data['thal'], prefix = "thal" , drop_first=True)

slope = pd.get_dummies(data['slope'], prefix = "slope", drop_first=True)
new_data = pd.concat([data, cp, thal, slope], axis=1)

new_data.head(3)
new_data.drop(['cp', 'thal', 'slope'], axis=1, inplace=True)

new_data.head(3)
X = new_data.drop(['target'], axis=1)

y = new_data.target
print(X.shape)
X = (X - X.min())/(X.max()-X.min())

X.head(3)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train, y_train)

lr.score(X_test, y_test)
from sklearn.model_selection import GridSearchCV
params = {'penalty':['l1','l2'],

         'C':[0.01,0.1,1,10,100],

         'class_weight':['balanced',None]}

lr_model = GridSearchCV(lr,param_grid=params,cv=10)
lr_model.fit(X_train,y_train)

lr_model.best_params_
lr = LogisticRegression(C=1, penalty='l2')

lr.fit(X_train, y_train)

lr.score(X_test, y_test)
#Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, lr.predict(X_test))

sns.heatmap(cm, annot=True)

plt.plot()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn.score(X_test, y_test)
for i in range(1,11):

    knn = KNeighborsClassifier()

    knn.fit(X_train, y_train)

    print("k : ",i ,"score : ",knn.score(X_test, y_test), end="\n" )
#Confusion Matrix

cm = confusion_matrix(y_test, knn.predict(X_test))

sns.heatmap(cm, annot=True)

plt.plot()
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=1)

dt.fit(X_train, y_train)

dt.score(X_test, y_test)
#Confusion Matrix

cm = confusion_matrix(y_test, dt.predict(X_test))

sns.heatmap(cm, annot=True)

plt.plot()
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()

gbc.fit(X_train, y_train)

gbc.score(X_test, y_test)
#Confusion Matrix

cm = confusion_matrix(y_test, gbc.predict(X_test))

sns.heatmap(cm, annot=True)

plt.plot()
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train, y_train)

nb.score(X_test, y_test)
#Confusion Matrix

cm = confusion_matrix(y_test, nb.predict(X_test))

sns.heatmap(cm, annot=True)

plt.plot()
from sklearn.ensemble import RandomForestClassifier

for i in range(1, 20):

    rfc = RandomForestClassifier(n_estimators=i)

    rfc.fit(X_train, y_train)

    print('estimators : ', i, "score : ", rfc.score(X_test, y_test), end="\n")
for i in range(1, 10):

    rfc = RandomForestClassifier(n_estimators=100, max_depth=i)

    rfc.fit(X_train, y_train)

    print('max_depth : ', i, "score : ", rfc.score(X_test, y_test), end="\n")
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc.score(X_test, y_test)
#Confusion Matrix

cm = confusion_matrix(y_test, rfc.predict(X_test))

sns.heatmap(cm, annot=True)

plt.plot()
from sklearn.svm import SVC

svc = SVC(kernel='linear')

svc.fit(X_train, y_train)

svc.score(X_test, y_test)
#Confusion Matrix

cm = confusion_matrix(y_test, svc.predict(X_test))

sns.heatmap(cm, annot=True)

plt.plot()
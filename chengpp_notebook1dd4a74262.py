import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from sklearn.model_selection import cross_val_score

print(check_output(["ls", "../input"]).decode("utf8"))
data=pd.read_csv('../input/xAPI-Edu-Data.csv')

data.shape
# check missing data

data.info()
data.columns
data['gender'].value_counts()
data['PlaceofBirth'].value_counts()
pd.crosstab(data['Class'],data['Topic'])
sns.factorplot(x="Relation", y="Class", data=data)
sns.boxplot(x="Class", y="raisedhands", data=data)

sns.swarmplot(x="Class", y="raisedhands", data=data, color=".15")
sns.boxplot(x="Class", y="VisITedResources", data=data)

sns.swarmplot(x="Class", y="VisITedResources", data=data, color=".25")
sns.boxplot(x="Class", y="AnnouncementsView", data=data)

sns.swarmplot(x="Class", y="AnnouncementsView", data= data, color=".25")
sns.boxplot(x="Class", y="Discussion", data=data)

sns.swarmplot(x="Class", y="Discussion", data= data, color=".25")
sns.factorplot(x="ParentAnsweringSurvey", y="Class", data=data)
sns.factorplot(x="ParentschoolSatisfaction", y="Class", data=data)
sns.factorplot(x="StudentAbsenceDays", y="Class", data=data)
data.dtypes
Features = data.drop('Class',axis=1)

Target = data['Class']

label = LabelEncoder()

Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index

for col in Cat_Colums:

    Features[col] = label.fit_transform(Features[col])
X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.25, random_state=52)
Logit_Model = LogisticRegression()

Logit_Model.fit(X_train,y_train)
Prediction = Logit_Model.predict(X_test)

Score = accuracy_score(y_test,Prediction)

Report = classification_report(y_test,Prediction)
print(Score)
print(Report)
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 300, random_state = 52)

forest.fit(X_train,y_train)

Prediction = forest.predict(X_test)

Score = accuracy_score(y_test,Prediction)

Report = classification_report(y_test,Prediction)
print(Score)
print(Report)
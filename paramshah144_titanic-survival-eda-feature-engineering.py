import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB



sns.set(rc={'figure.figsize':(12, 10)})
data = pd.read_csv('../input/titanic/train.csv')
data.head(10)
data.info()
data.isnull().sum()
data.describe()


plt.figure(figsize=(12, 10))

heatmap = sns.heatmap(data[["Survived","SibSp","Parch","Age","Fare","Sex"]].corr(), annot=True)
data['SibSp'].nunique()
data['SibSp'].unique()
bargraph_sibsp = sns.factorplot(x = "SibSp", y = "Survived", data = data, kind = "bar", size = 8)

bargraph_sibsp = bargraph_sibsp.set_ylabels("survival probability")
age_visual = sns.FacetGrid(data, col = 'Survived', size=7)

age_visual = age_visual.map(sns.distplot, "Age")

age_visual = age_visual.set_ylabels("survival probability")
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))

age_plot = sns.barplot(x = "Sex",y = "Survived", data = data)

age_plot = age_plot.set_ylabel("Survival Probability")
data[["Sex","Survived"]].groupby('Sex').mean()
pclass = sns.factorplot(x = "Pclass", y = "Survived", data = data, kind = "bar", size = 8)

pclass = pclass.set_ylabels("survival probability")
g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", data=data, size=6, kind="bar")

g = g.set_ylabels("survival probability")
data["Embarked"].isnull().sum()
data["Embarked"].value_counts()
#Fill Embarked with 'S' i.e. the most frequent values

data["Embarked"] = data["Embarked"].fillna("S")
g = sns.factorplot(x="Embarked", y="Survived", data=data, size=7, kind="bar")

g = g.set_ylabels("survival probability")
# Explore Pclass vs Embarked 

g = sns.factorplot("Pclass", col="Embarked",  data=data, size=7, kind="count")

g.despine(left=True)

g = g.set_ylabels("Count")
pd.read_csv('../input/titanic/test.csv')
data.head()
data.info()
mean = data["Age"].mean()

std = data["Age"].std()

is_null = data["Age"].isnull().sum()



rand_age = np.random.randint(mean - std, mean + std, size = is_null)

    



age_slice = data["Age"].copy()

age_slice[np.isnan(age_slice)] = rand_age

data["Age"] = age_slice
data["Age"].isnull().sum()
data.info()
data["Embarked"].isnull().sum()
#Fill Embarked with 'S' i.e. the most frequent values

data["Embarked"] = data["Embarked"].fillna("S")
col_to_drop = ['PassengerId','Cabin', 'Ticket','Name']

data.drop(col_to_drop, axis=1, inplace = True)
data.head()
genders = {"male": 0, "female": 1}

data['Sex'] = data['Sex'].map(genders)
data.head()
ports = {"S": 0, "C": 1, "Q": 2}



data['Embarked'] = data['Embarked'].map(ports)
data.head()
data.info()
# input and output data



x = data.drop(data.columns[[0]], axis = 1)

y = data['Survived']
x.head()
y.head()
# splitting into training and testing data

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state =0)
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

xtrain = sc_x.fit_transform(xtrain) 

xtest = sc_x.transform(xtest)
logreg = LogisticRegression()

svc_classifier = SVC()

dt_classifier = DecisionTreeClassifier()

knn_classifier = KNeighborsClassifier(5)

rf_classifier = RandomForestClassifier(n_estimators=1000)

naivebayes_classifier = GaussianNB()
logreg.fit(xtrain, ytrain)

svc_classifier.fit(xtrain, ytrain)

dt_classifier.fit(xtrain, ytrain)

knn_classifier.fit(xtrain, ytrain)

rf_classifier.fit(xtrain, ytrain)

naivebayes_classifier.fit(xtrain, ytrain)
logreg_ypred = logreg.predict(xtest)

svc_classifier_ypred = svc_classifier.predict(xtest)

dt_classifier_ypred = dt_classifier.predict(xtest)

knn_classifier_ypred = knn_classifier.predict(xtest)

rf_classifier_ypred = rf_classifier.predict(xtest)

naivebayes_y_pred = naivebayes_classifier.predict(xtest)


from sklearn.metrics import accuracy_score



logreg_acc = accuracy_score(ytest, logreg_ypred)

svc_classifier_acc = accuracy_score(ytest, svc_classifier_ypred)

dt_classifier_acc = accuracy_score(ytest, dt_classifier_ypred)

knn_classifier_acc = accuracy_score(ytest, knn_classifier_ypred)

rf_classifier_acc = accuracy_score(ytest, rf_classifier_ypred)

naivebayes_acc = accuracy_score(ytest, naivebayes_y_pred)
print ("Logistic Regression : ", round(logreg_acc*100, 2))

print ("Support Vector      : ", round(svc_classifier_acc*100, 2))

print ("Decision Tree       : ", round(dt_classifier_acc*100, 2))

print ("K-NN Classifier     : ", round(knn_classifier_acc*100, 2))

print ("Random Forest       : ", round(rf_classifier_acc*100, 2))

print ("Naive Bayes         : ", round(naivebayes_acc*100, 2))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, naivebayes_y_pred)
cm

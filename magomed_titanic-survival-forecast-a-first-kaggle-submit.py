import numpy as np # linear algebra

import pandas as pd # data processing libary

from sklearn import preprocessing, svm, tree, linear_model, metrics #methods 

#for machine learning tools like preprocessing, models and metrics

from sklearn.model_selection import train_test_split #replace cross_validation method

import matplotlib.pyplot as plt #plotting library

%matplotlib inline
#first let's read the data and take a look

titanic_df = pd.read_csv("../input/train.csv") 

titanic_df.head()
titanic_df.info()
print("Cabin value count: %d " % titanic_df["Cabin"].count())

print("Age value count: %d" % titanic_df["Age"].count())

print("Embarked value count: %d" % titanic_df["Embarked"].count())
titanic_df.drop("Cabin",axis=1,inplace=True)

titanic_df["Embarked"].value_counts()
titanic_df["Embarked"].fillna("S",inplace = True)

titanic_df.dropna(inplace=True)

titanic_df.info()
print("Name value count: %d " % titanic_df["Name"].value_counts().size)

print("Ticket value count: %d " % titanic_df["Ticket"].value_counts().size)

print("PassengerId value count: %d " % titanic_df["PassengerId"].value_counts().size)

print("Sex value count: %d " % titanic_df["Sex"].value_counts().size)

print("Embarked value count: %d " % titanic_df["Embarked"].value_counts().size)
titanic_df.drop(["Name","Ticket","PassengerId"],axis=1, inplace=True)

sex_labels= titanic_df["Sex"].unique()

embarked_labels = titanic_df["Embarked"].unique()
le = preprocessing.LabelEncoder()

le.fit(titanic_df.Sex.values)

titanic_df["Sex"] = le.transform(titanic_df.Sex.values)

sex_labels = titanic_df["Sex"].unique()

sex_labelsE = le.inverse_transform(sex_labels)

le.fit(titanic_df.Embarked.values)

titanic_df["Embarked"] = le.transform(titanic_df.Embarked.values)

embarked_labels = titanic_df["Embarked"].unique()

embarked_labelsE = le.inverse_transform(embarked_labels)
titanic_df.head()
fig = plt.figure()

ax = fig.add_subplot(111)

titanic_df.groupby('Pclass').sum()['Survived'].plot.pie(

    figsize = (8,8), autopct = '%1.1f%%', startangle = 90, fontsize = 15, explode=(0.05,0,0) )

ax.set_ylabel('')

ax.set_title('Survival rate', fontsize = 16)

ax.legend(labels = titanic_df['Pclass'].unique().sort(), loc = "best", title='Class', fontsize=14)
fig = plt.figure()

ax = fig.add_subplot(111)

ax.set_ylabel("Survival rate")

titanic_df.groupby("Pclass").mean()["Survived"].plot.bar()

ax.set_xticklabels(labels = ax.get_xticklabels(),rotation=0)
fig = plt.figure()

sorted_labes = [x for (y,x) in sorted(zip(sex_labels,sex_labelsE))]

ax = fig.add_subplot(111)

ax.set_ylabel("Survival rate")

titanic_df.groupby("Sex").mean()["Survived"].plot.bar()

ax.set_xticklabels(labels = sorted_labes,rotation=20)
index_name=titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"].index.names

index_level=titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"].index.levels

index_ = zip(index_name,index_level)
fig, axes = plt.subplots(nrows=1, ncols=3)

titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"][1].plot.bar(ax=axes[0] )

titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"][2].plot.bar(ax=axes[1] )

titanic_df.groupby(["Pclass","Sex"]).mean()["Survived"][3].plot.bar(ax=axes[2] )

axes[0].set_title('Class 1')

axes[0].set_xticklabels(labels = sorted_labes,rotation=20)

axes[0].set_yticks(np.arange(0.0,1.1,0.1))

axes[1].set_title('Class 2')

axes[1].set_xticklabels(labels = sorted_labes,rotation=20)

axes[1].set_yticks(np.arange(0.0,1.1,0.1))

axes[2].set_title('Class 3')

axes[2].set_xticklabels(labels = sorted_labes,rotation=20)

axes[2].set_yticks(np.arange(0.0,1.1,0.1))

fig.tight_layout()
years_range = np.arange(0,90,10)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(8,12))

titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][1,0].plot.bar(ax=axes[0,0], title = ("Women Class 1") )

titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][1,1].plot.bar(ax=axes[0,1], title = ("Men Class 1") )

titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][2,0].plot.bar(ax=axes[1,0], title = ("Women Class 2") )

titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][2,1].plot.bar(ax=axes[1,1], title = ("Men Class 2") )

titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][3,0].plot.bar(ax=axes[2,0], title = ("Women Class 3") )

titanic_df.groupby(by=["Pclass","Sex",pd.cut(titanic_df["Age"],years_range)]).mean()["Survived"][3,1].plot.bar(ax=axes[2,1], title = ("Men Class 3") )

axes[0,0].set_yticks(np.arange(0.0,1.1,0.1))

axes[0,1].set_yticks(np.arange(0.0,1.1,0.1))

axes[1,0].set_yticks(np.arange(0.0,1.1,0.1))

axes[1,1].set_yticks(np.arange(0.0,1.1,0.1))

axes[2,0].set_yticks(np.arange(0.0,1.1,0.1))

axes[2,1].set_yticks(np.arange(0.0,1.1,0.1))

fig.tight_layout()
fig, axes = plt.subplots(nrows=1, ncols=3)

sorted_labes = [x for (y,x) in sorted(zip(embarked_labels,embarked_labelsE))]

titanic_df.groupby(["Pclass","Embarked"]).mean()["Survived"][1].plot.bar(ax=axes[0] )

titanic_df.groupby(["Pclass","Embarked"]).mean()["Survived"][2].plot.bar(ax=axes[1] )

titanic_df.groupby(["Pclass","Embarked"]).mean()["Survived"][3].plot.bar(ax=axes[2] )

axes[0].set_title('Class 1')

axes[0].set_yticks(np.arange(0.0,1.1,0.1))

axes[0].set_xticklabels(labels = sorted_labes,rotation=20)

axes[1].set_title('Class 2')

axes[1].set_yticks(np.arange(0.0,1.1,0.1))

axes[1].set_xticklabels(labels = sorted_labes,rotation=20)

axes[2].set_title('Class 3')

axes[2].set_yticks(np.arange(0.0,1.1,0.1))

axes[2].set_xticklabels(labels = sorted_labes,rotation=20)

fig.tight_layout()
titanic_df.groupby("SibSp").mean()["Survived"].plot.bar()
titanic_df.groupby("Parch").mean()["Survived"].plot.bar()
fare_ranges = np.arange(0,max(titanic_df.Fare)+1,max(titanic_df.Fare)/10)

titanic_df.groupby(pd.cut(titanic_df["Fare"],fare_ranges)).mean()["Survived"].plot.bar()
titanic_features = titanic_df.drop("Survived", axis=1)

feat_labels = titanic_df.columns[1:]
from sklearn import ensemble

forest = ensemble.RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

forest.fit(titanic_features,titanic_df["Survived"])

importances = forest.feature_importances_

indices= np.argsort(importances)[::-1]

for f in range(titanic_features.shape[1]):

    print("%2d) %-*s %f" % (f+1, 30, feat_labels[f], importances[indices[f]]))
titanic_3features = titanic_features[titanic_features.columns[:3]]

titanic_3features.head()
from sklearn import model_selection

from sklearn import preprocessing, metrics
#let's standarize the feature value to improve the prediction

sc = preprocessing.StandardScaler()

#------ for all features

sc.fit(titanic_features)

titanic_features_std = sc.transform(titanic_features)

#------ only 3 features

sc.fit(titanic_3features)

titanic_3features_std = sc.transform(titanic_3features)
#let's split the data into training and test subsets

#-------- for all features

x_train, x_test, y_train, y_test =  model_selection.train_test_split(

    titanic_features_std, titanic_df.Survived, test_size = 0.3, random_state = 0)

#-------- only 3 features

x_3f_train, x_3f_test, y_3f_train, y_3f_test = model_selection.train_test_split(

    titanic_3features_std, titanic_df.Survived, test_size = 0.3, random_state = 0)
from sklearn.tree import DecisionTreeClassifier

cm_tree = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=0)
cm_tree.fit(x_train,y_train)
y_predict = cm_tree.predict(x_test)

print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))

print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))
cm_tree.fit(x_3f_train,y_3f_train)
y_3f_predict = cm_tree.predict(x_3f_test)

print("The accuracy is: %2f" % metrics.accuracy_score(y_3f_test,y_3f_predict))

print("The precision is: %2f" % metrics.precision_score(y_3f_test,y_3f_predict))
from sklearn.linear_model import LogisticRegression

cm_lr = LogisticRegression(C=1000.0, random_state = 0)
cm_lr.fit(x_train,y_train)
y_predict = cm_lr.predict(x_test)

print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))

print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))
cm_lr.fit(x_3f_train,y_3f_train)
y_predict = cm_lr.predict(x_3f_test)

print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))

print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))
from sklearn.svm import SVC

svm = SVC(kernel = 'linear', C = 10.0, random_state = 0)
svm.fit(x_train, y_train)
y_predict = svm.predict(x_test)

print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))

print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))
svm.fit(x_3f_train, y_3f_train)
y_predict = svm.predict(x_3f_test)

print("The accuracy is: %2f" % metrics.accuracy_score(y_test,y_predict))

print("The precision is: %2f" % metrics.precision_score(y_test,y_predict))
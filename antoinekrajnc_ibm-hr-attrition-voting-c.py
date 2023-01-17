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
dataset = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")

dataset.head()
dataset.Attrition = dataset.Attrition.apply(lambda x: 0 if x=="No" else 1)
print(np.unique(dataset.BusinessTravel), len(np.unique(dataset.BusinessTravel)))

print(np.unique(dataset.Department), len(np.unique(dataset.Department)))

print(np.unique(dataset.EducationField), len(np.unique(dataset.EducationField)))

print(np.unique(dataset.JobRole), len(np.unique(dataset.JobRole)))

print(np.unique(dataset.MaritalStatus), len(np.unique(dataset.MaritalStatus)))
dataset.isnull().any()
X = pd.get_dummies(dataset.loc[:, dataset.columns!="Attrition"], drop_first=True)

y = pd.get_dummies(dataset.Attrition, drop_first=True)
import seaborn as sns

sns.countplot(x=dataset.Attrition, palette="hls")
sns.catplot(x="Gender", y="Attrition", hue="JobRole", data=dataset, kind="bar")
sns.catplot(x="Gender", y="Attrition", data=dataset, kind="bar")
sns.countplot(x="Gender", data=dataset)
sns.catplot(x="Gender", y="Attrition", hue="BusinessTravel", data=dataset, kind="bar")
sns.catplot(x="Gender", y="Attrition", hue="Department", data=dataset, kind="bar")
sns.catplot(x="Gender", y="Attrition", hue="EducationField", data=dataset, kind="bar")
sns.catplot(x="Gender", y="Attrition", hue="MaritalStatus", data=dataset, kind="bar")
col_to_drop = []

for x in X:

    if len(np.unique(X[x])) == 1:

        col_to_drop+=[x]
X.drop(col_to_drop, axis=1)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier().fit(X,y)

from sklearn.tree import export_graphviz

import graphviz 

viz = export_graphviz(dt,

                feature_names=X.columns,

                filled=True,

                rounded=True,

                max_depth=5,

                )

graph = graphviz.Source(viz) 

graph
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X = sc_x.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size =0.3, stratify=y)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import BernoulliNB
random_state = 40

classifiers = []

classifiers.append(SVC(random_state=random_state))

classifiers.append(DecisionTreeClassifier(random_state=random_state))

classifiers.append(KNeighborsClassifier())

classifiers.append(LogisticRegression(random_state = random_state))

classifiers.append(RandomForestClassifier(random_state=random_state))

classifiers.append(BernoulliNB())
from sklearn.model_selection import cross_val_score

cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, 

                                      X_train, 

                                      np.ravel(y_train), 

                                      scoring = "accuracy", 

                                      cv = 10, 

                                      n_jobs=4))

    

cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC",

                                                                                      "DecisionTree",

                                                                                      "KNeighboors",

                                                                                      "LogisticRegression",

                                                                                      "RandomForest",

                                                                                      "Naive Bayes"]})

                                                                                      



g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res.sort_values("CrossValMeans")
votingC = VotingClassifier(estimators=[                                       

                                       ("LogReg",classifiers[3]), 

                                       ("RF",classifiers[4]),

                                       ("Naive Bayes", classifiers[5])], 

                           voting='hard', 

                           n_jobs=4)
votingC.fit(X_train, y_train)
cross_val_score(votingC, X_train, np.ravel(y_train), scoring = "accuracy", cv = 10, n_jobs=4).mean()
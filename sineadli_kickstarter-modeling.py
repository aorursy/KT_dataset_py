# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
kick = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")

# narrowing with conditions
kick = kick[(kick["state"] == "successful") | (kick["state"] == "failed")]
kick = kick[(kick["goal"] <= 50000) & (kick["currency"] == "USD") & (kick["country"] == "US")]

# creating the campaign_days feature
kick["campaign_days"] = pd.to_datetime(kick["deadline"]) - pd.to_datetime(kick["launched"])
kick["campaign_days"] = kick["campaign_days"].dt.days + 1

# dropping features
kick = kick.drop(["currency", "country", "usd pledged", "usd_pledged_real", "usd_goal_real"], axis=1)

kick.head()
kick = kick[["ID", "main_category", "goal", "campaign_days", "state"]]
kick.head()
kick.loc[kick["state"] == "successful", "state"] = 1
kick.loc[kick["state"] == "failed", "state"] = 0
kick["state"] = kick["state"].astype("int")
kick["main_category"] = kick["main_category"].astype("category")
kick["main_category"] = kick["main_category"].cat.codes
import seaborn as sns
sns.distplot(kick["goal"])
kick["goal_by_5000"] = pd.cut(x=kick["goal"], bins=[0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000], labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
kick.head()
kick = kick.drop("goal", axis=1)
kick["goal_by_5000"] = kick["goal_by_5000"].astype("int")
kick["campaign_days_by_19"] = pd.cut(x=kick["campaign_days"], bins=[0,19,38,57,76,95], labels=[1, 2, 3, 4, 5])
kick = kick.drop("campaign_days", axis=1)
kick["campaign_days_by_19"] = kick["campaign_days_by_19"].astype("int")
kick.head()
kick.dtypes
from sklearn.model_selection import train_test_split

train, test = train_test_split(kick, test_size=0.3, random_state=42)
X_train = train.drop("state", axis=1)
Y_train = train["state"]
X_test  = test.drop("ID", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
from sklearn.svm import SVC, LinearSVC

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)
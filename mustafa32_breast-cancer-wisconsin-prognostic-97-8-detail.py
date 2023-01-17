import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/breast-cancer-wisconsin-prognostic-data-set/data 2.csv")
df.head()
df.isnull().sum()
sns.lineplot(x=df["radius_mean"],y=df["perimeter_mean"], hue=df["diagnosis"])
sns.countplot(df['diagnosis'])
sns.barplot(df['diagnosis'],df['area_mean'])
sns.scatterplot(x = df['area_mean'],y= df['smoothness_mean'],hue=df['diagnosis'])
sns.regplot(x = df['area_mean'],y= df['smoothness_mean'])
sns.lmplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=df)
sns.swarmplot(x=df['diagnosis'],y=df['smoothness_mean'])
sns.distplot(df['perimeter_mean'])
sns.distplot(df['smoothness_mean'])
sns.jointplot(df['perimeter_mean'],df['smoothness_mean'],kind='kde')
df.head()
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

X = df.drop(['id','Unnamed: 32','diagnosis'],axis=1)

y = df['diagnosis']



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)





model = [["LogisticRegression",LogisticRegression()],["RandomForestClassifier",RandomForestClassifier()],["DecisionTreeClassifier",DecisionTreeClassifier()],["GaussianNB",GaussianNB()],["KNeighborsClassifier",KNeighborsClassifier()]]
accuracy_score1 = []

for i in model:

    log = i[1]

    log.fit(X_train,y_train)

    predict = log.predict(X_test)

    accuracy_score1.append([i[0],accuracy_score(predict,y_test)])    
main_score = pd.DataFrame(accuracy_score1)

main_score.columns = ["Model","Score"]
main_score
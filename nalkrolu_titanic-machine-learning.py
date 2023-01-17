import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,learning_curve

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import tree

from sklearn.cluster import KMeans

from xgboost import XGBClassifier,XGBRFClassifier

from sklearn.metrics import confusion_matrix,accuracy_score
from yellowbrick.classifier import ROCAUC

from yellowbrick.classifier import ConfusionMatrix
from pylab import rcParams

rcParams['figure.figsize'] = 14,6

plt.style.use('seaborn-talk')

#plt.style.available
data = pd.read_csv('/kaggle/input/titanic2/titanic.csv')

data.head()
data.info()
sns.countplot(x="survived",data=data)

plt.show()
sns.countplot(x="survived",hue="pclass",data=data)

plt.show()
sns.countplot(x="survived",hue="sex",data=data)

plt.show()
plt.subplot(2,2,1)

sns.distplot(data.fare)

plt.subplot(2,2,2)

sns.distplot(data.age)

plt.show()
sns.countplot(x="sibsp",data=data,hue="survived")

plt.show()
sns.countplot(x="parch",data=data,hue="survived")

plt.show()
plt.subplot(2,2,1)

sns.violinplot(x="survived",y="age",data=data)

plt.subplot(2,2,2)

sns.violinplot(x="survived",y="fare",data=data)

plt.show()
data.isnull().sum()
sns.heatmap(data.isnull(),yticklabels=False)

plt.show()
data.drop(["cabin","boat","body","home.dest"],axis=1,inplace=True)

data.head()
data.isnull().sum()
data.dropna(inplace=True)

data.isnull().sum()
sex=pd.get_dummies(data["sex"],drop_first=True)

sex.head()
data.embarked.value_counts()
embarked = pd.get_dummies(data["embarked"],drop_first=True)

pclass=pd.get_dummies(data["pclass"],drop_first=True)

data.drop(["sex","embarked","pclass"],axis=1,inplace=True)

data = pd.concat([data,sex,embarked,pclass],axis=1)

data.info()
data.drop(["name","ticket"],axis=1,inplace=True)

data.info()
plt.figure(figsize=(10,4))

sns.scatterplot(x="age",y="survived",data=data,size="fare")

plt.title("Age vs Survived (Fare)")

plt.legend(loc=7);plt.show()
X=data.drop("survived",axis=1)

y=data["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.25,random_state=100)
lr = LogisticRegression(solver="lbfgs",max_iter=500)

lr.fit(X_train,y_train)

score = lr.score(X_test,y_test)

pred_lr= lr.predict(X_test)



print("=LogisticRegression=")

print("Accuracy Score:",accuracy_score(y_test,pred_lr))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(lr, classes=y)

visualizer.fit(X_train, y_train)       # Fit the training data to the visualizer

visualizer.score(X_test, y_test)       # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(lr, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
knc = KNeighborsClassifier(n_neighbors=24)

knc.fit(X_train,y_train)

score = knc.score(X_test,y_test)

pred_knc= knc.predict(X_test)



print("=KNeighborsClassifier=")

print("Accuracy Score:",accuracy_score(y_test,pred_knc))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(knc, classes=y)

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(knc, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
net = MLPClassifier(max_iter=1000)

net.fit(X_train,y_train)

score = net.score(X_test,y_test)

pred_net= net.predict(X_test)



print("=MLPClassifier=")

print("Accuracy Score:",accuracy_score(y_test,pred_net))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(net, classes=y)

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(net, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
dt = tree.DecisionTreeClassifier()

dt.fit(X_train,y_train)

score = dt.score(X_test,y_test)

pred_dt= dt.predict(X_test)



print("=DecisionTreeClassifier=")

print("Accuracy Score:",accuracy_score(y_test,pred_dt))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(dt, classes=y)

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(dt, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
gbc = GradientBoostingClassifier(n_estimators=10,random_state=0)

gbc.fit(X_train,y_train)

score = gbc.score(X_test,y_test)

pred_gbc= gbc.predict(X_test)



print("=GradientBoostingClassifier=")

print("Accuracy Score:",accuracy_score(y_test,pred_gbc))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(gbc, classes=y)

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(gbc, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
xg = XGBRFClassifier(learning_rate=0.0001,objective="binary:logistic").fit(X_train,y_train)

xg.score(X_test,y_test)

score = xg.score(X_test,y_test)

pred_xg= xg.predict(X_test)



print("=XGBRFClassifier=")

print("Accuracy Score:",accuracy_score(y_test,pred_xg))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(xg, classes=y)

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(xg, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
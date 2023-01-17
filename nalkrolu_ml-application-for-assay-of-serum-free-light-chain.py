import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,learning_curve

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.cluster import KMeans

from xgboost import XGBClassifier,XGBRFClassifier

from sklearn.metrics import confusion_matrix,accuracy_score
plt.style.use('seaborn-talk')

#plt.style.available
from pylab import rcParams

rcParams['figure.figsize'] = 14,6
data = pd.read_csv('/kaggle/input/assay-of-serum-free-light-chain/flchain.csv').sort_values("sample.yr").reset_index()

data.drop(["Unnamed: 0","index"],axis=1,inplace=True)
data.head()
sns.scatterplot(x="kappa" ,y="lambda" ,data=data,hue="flc.grp",palette="coolwarm")

plt.show()
sns.scatterplot(x="futime" ,y="age",data=data,hue="death",palette="coolwarm")

plt.show()
data.shape
data["sex"].replace(to_replace=["M","F"],value=[1,0],inplace=True)
plt.style.use('classic')

rcParams['figure.figsize'] = 14,6

sns.heatmap(data.isnull(),yticklabels=False,cmap="coolwarm");
sns.heatmap(data.corr(),annot=True,cmap="coolwarm")

plt.show()
data.isnull().sum()
data["creatinine"].fillna("?",inplace=True)

plt.hist(data.loc[data["creatinine"]!="?"]["creatinine"],alpha=0.5,density=True,bins=100)

plt.title("Histogram of Creatinine")

plt.show()
data["chapter"].value_counts()
sns.countplot(y="chapter",data=data,palette="dark")

plt.show()
data.drop(["creatinine","chapter"],axis=1,inplace=True)
data.head()
data.isnull().sum()
sns.countplot(x="mgus",hue="sex",data=data)

plt.legend({"Female","Male"})

plt.show()
sns.countplot(x="death",hue="sex",data=data)

plt.legend({"Female","Male"})

plt.show()
sns.countplot(x="sex",data=data)

plt.xticks([0,1],["Female","Male"])

plt.show()
sns.countplot(x="flc.grp",data=data)

plt.show()
sns.countplot(x="sample.yr",data=data)

plt.show()
plt.subplot(1,2,1)

plt.hist(data.loc[data["death"]==1]["age"],alpha=0.7,label="Death:1",density=True,bins=20)

plt.hist(data.loc[data["death"]==0]["age"],alpha=0.7,label="Death:0",density=True,bins=20)

plt.legend()

plt.title("age")



plt.subplot(1,2,2)

plt.hist(data.loc[data["death"]==1]["kappa"],alpha=0.7,label="Death:1",density=True,bins=30)

plt.hist(data.loc[data["death"]==0]["kappa"],alpha=0.7,label="Death:0",density=True,bins=30)

plt.title("kappa")

plt.legend()

plt.show()
plt.subplot(1,2,1)

plt.hist(data.loc[data["death"]==1]["lambda"],alpha=0.7,label="Death:1",density=True,bins=30)

plt.hist(data.loc[data["death"]==0]["lambda"],alpha=0.7,label="Death:0",density=True,bins=30)

plt.legend()

plt.title("lambda")



plt.subplot(1,2,2)

plt.hist(data.loc[data["death"]==1]["futime"],alpha=0.7,label="Death:1",density=True,bins=30)

plt.hist(data.loc[data["death"]==0]["futime"],alpha=0.7,label="Death:0",density=True,bins=30)

plt.title("futime")

plt.legend()

plt.show()
plt.subplot(1,2,1)

sns.boxplot(x="death",y="age",data=data);



plt.subplot(1,2,2)

sns.boxplot(x="sex",y="age",data=data);

plt.xticks([0,1],["Female","Male"])

plt.show()
from yellowbrick.classifier import ROCAUC

from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.classifier import DiscriminationThreshold
X = data.drop("death",axis=1) 

y = data["death"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.35,random_state=42)
lr = LogisticRegression(solver="lbfgs",max_iter=500)

lr.fit(X_train,y_train)

score = lr.score(X_test,y_test)

pred_lr= lr.predict(X_test)



print("=LogisticRegression=")

print("Test Variable Score:",score)

print("Accuracy Score     :",accuracy_score(y_test,pred_lr))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(lr, classes=[0,1])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(lr, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
gnb = GaussianNB()

gnb.fit(X_train,y_train)

score = gnb.score(X_test,y_test)

pred_knc= gnb.predict(X_test)



print("=GaussianNB=")

print("Test Variable Score:",score)

print("Accuracy Score     :",accuracy_score(y_test,pred_knc))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(gnb, classes=[0,1])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(gnb, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
knc = KNeighborsClassifier(n_neighbors=24)

knc.fit(X_train,y_train)

score = knc.score(X_test,y_test)

pred_knc= knc.predict(X_test)



print("=KNeighborsClassifier=")

print("Test Variable Score:",score)

print("Accuracy Score     :",accuracy_score(y_test,pred_knc))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(knc, classes=[0,1])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(knc, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
net = MLPClassifier(max_iter=1000,activation="logistic")

net.fit(X_train,y_train)

score = net.score(X_test,y_test)

pred_net= net.predict(X_test)



print("=MLPClassifier=")

print("Test Variable Score:",score)

print("Accuracy Score     :",accuracy_score(y_test,pred_net))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(net, classes=[0,1])

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

print("Test Variable Score:",score)

print("Accuracy Score     :",accuracy_score(y_test,pred_dt))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(dt, classes=[0,1])

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

print("Test Variable Score:",score)

print("Accuracy Score     :",accuracy_score(y_test,pred_gbc))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(gbc, classes=[0,1])

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

print("Test Variable Score:",score)

print("Accuracy Score     :",accuracy_score(y_test,pred_xg))



plt.figure(figsize=(10,5))

visualizer = ROCAUC(xg, classes=[0,1])

visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer

visualizer.score(X_test, y_test)        # Evaluate the model on the test data

visualizer.show();



plt.figure(figsize=(5,4))

cm = ConfusionMatrix(xg, classes=[0,1])

cm.fit(X_train, y_train)

cm.score(X_test, y_test)

cm.show();
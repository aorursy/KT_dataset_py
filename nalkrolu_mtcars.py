import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('classic')
from sklearn.model_selection import train_test_split,learning_curve

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import tree

from sklearn.svm import SVC

from sklearn.cluster import KMeans

from xgboost import XGBClassifier,XGBRFClassifier

from sklearn.metrics import confusion_matrix,accuracy_score
from pylab import rcParams

rcParams['figure.figsize'] = 14,6
data = pd.read_csv("/kaggle/input/mtcars/mtcars.csv")

data.index= data["Unnamed: 0"]

data = data.drop("Unnamed: 0",axis=1)

data.head()
data.isnull().sum()
data.info()
data.describe()
sns.countplot(x="vs",hue="am",data=data)

plt.show()
sns.countplot(x="cyl",hue="am",data=data)

plt.show()
sns.countplot(x="gear",hue="am",data=data)

plt.show()
sns.countplot(x="carb",hue="am",data=data)

plt.show()
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

plt.barh(data.index,data.wt)

plt.xticks(rotation=80)

plt.rcParams["xtick.labelsize"]=9

plt.title("Weight (1000 lbs)")

plt.subplot(2,2,2)

plt.barh(data.index,data.qsec)

plt.xticks(rotation=80)

plt.rcParams["xtick.labelsize"]=9

plt.title("1/4 mile time")

plt.subplot(2,2,3)

plt.barh(data.index,data.drat)

plt.xticks(rotation=80)

plt.rcParams["xtick.labelsize"]=9

plt.title("Rear axle ratio")

plt.subplot(2,2,4)

plt.barh(data.index,data.mpg)

plt.xticks(rotation=80)

plt.rcParams["xtick.labelsize"]=9

plt.title("Miles/(US) gallon")

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,7))

plt.subplot(2,2,1)

sns.violinplot(x="am",y="mpg",data=data)

sns.swarmplot(x="am",y="mpg",data=data,color="0000")

plt.subplot(2,2,2)

sns.violinplot(x="am",y="wt",data=data)

sns.swarmplot(x="am",y="wt",data=data,color="0000")

plt.subplot(2,2,3)

sns.violinplot(x="am",y="drat",data=data)

sns.swarmplot(x="am",y="drat",data=data,color="0000")

plt.subplot(2,2,4)

sns.violinplot(x="am",y="disp",data=data)

sns.swarmplot(x="am",y="disp",data=data,color="0000")

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,7))

plt.subplot(2,2,1)

data.loc[data["am"]==1]["mpg"].hist(bins=10,alpha=0.7,label="am:1")

data.loc[data["am"]==0]["mpg"].hist(bins=10,alpha=0.7,label="am:0")

plt.title("Histogram of Miles/(US) gallon")

plt.legend()

plt.subplot(2,2,2)

data.loc[data["am"]==1]["wt"].hist(bins=10,alpha=0.7,label="am:1")

data.loc[data["am"]==0]["wt"].hist(bins=10,alpha=0.7,label="am:0")

plt.title("Histogram of Weight (1000 lbs)")

plt.legend()

plt.subplot(2,2,3)

data.loc[data["am"]==1]["drat"].hist(bins=10,alpha=0.7,label="am:1")

data.loc[data["am"]==0]["drat"].hist(bins=10,alpha=0.7,label="am:0")

plt.title("Histogram of Rear axle ratio")

plt.legend()

plt.subplot(2,2,4)

data.loc[data["am"]==1]["disp"].hist(bins=10,alpha=0.7,label="am:1")

data.loc[data["am"]==0]["disp"].hist(bins=10,alpha=0.7,label="am:0")

plt.title("Histogram of Displacement (cu.in.)")

plt.legend()

plt.tight_layout()

plt.show()
plt.figure(figsize=(15,7))

plt.subplot(2,2,1)

sns.scatterplot(x="wt",y="mpg",hue="am",size="hp",data=data,palette="Greens")

plt.xlabel("wt")

plt.ylabel("mpg")

plt.subplot(2,2,2)

sns.scatterplot(x="drat",y="wt",hue="am",size="hp",data=data,palette="Reds")

plt.xlabel("drat")

plt.ylabel("wt")

plt.subplot(2,2,3)

sns.scatterplot(x="disp",y="drat",hue="am",size="hp",data=data,palette="Greys")

plt.xlabel("disp")

plt.ylabel("drat")

plt.subplot(2,2,4)

sns.scatterplot(x="mpg",y="disp",hue="am",size="hp",data=data,palette="Wistia")

plt.xlabel("mpg")

plt.ylabel("disp")

plt.tight_layout()

plt.show()
sns.heatmap(data.corr(),annot=True)

plt.show()
from yellowbrick.classifier import ROCAUC

from yellowbrick.classifier import ConfusionMatrix

from yellowbrick.classifier import DiscriminationThreshold
X = data.drop("am",axis=1) 

y = data["am"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.40,random_state=42)
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
xg = XGBRFClassifier(learning_rate=0.01,objective="binary:logistic").fit(X_train,y_train)

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
from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression,Lasso,Ridge

from yellowbrick.regressor import PredictionError

from yellowbrick.regressor import ResidualsPlot

from yellowbrick.regressor import CooksDistance

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
X_reg = data.drop("hp",axis=1) 

y_reg = data["hp"]

X_train,X_test,y_train,y_test = train_test_split(X_reg,y_reg,test_size=.35,random_state=42)
plt.figure(figsize=(14,6))

visualizer = CooksDistance()

visualizer.fit(X_reg, y_reg)

visualizer.show();
lr = LinearRegression()

lr.fit(X_train,y_train)

print("=====LinearRegression=====")

print("Score Test:",lr.score(X_test,y_test))

pred_lr = lr.predict(X_test)



print("MAE:",mean_absolute_error(y_test,pred_lr))

print("MSE:",mean_squared_error(y_test,pred_lr))





visualizer = PredictionError(lr)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show();   



plt.figure(figsize=(12,6))

visualizer = ResidualsPlot(lr)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
net = MLPRegressor(hidden_layer_sizes=(100,),max_iter=30000,activation="logistic")

net.fit(X_train,y_train)

print("=====MLPRegressor=====")

print("Score Test:",net.score(X_test,y_test))

pred_net = net.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred_net))

print("MSE:",mean_squared_error(y_test,pred_net))



visualizer = PredictionError(net)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show();  



plt.figure(figsize=(12,6))

visualizer = ResidualsPlot(net)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
lr = Lasso()

lr.fit(X_train,y_train)

print("=====Lasso=====")

print("Score Test:",lr.score(X_test,y_test))

pred_lr = lr.predict(X_test)



print("MAE:",mean_absolute_error(y_test,pred_lr))

print("MSE:",mean_squared_error(y_test,pred_lr))





visualizer = PredictionError(lr)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show();   



plt.figure(figsize=(12,6))

visualizer = ResidualsPlot(lr)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
lr = Ridge()

lr.fit(X_train,y_train)

print("=====Ridge=====")

print("Score Test:",lr.score(X_test,y_test))

pred_lr = lr.predict(X_test)



print("MAE:",mean_absolute_error(y_test,pred_lr))

print("MSE:",mean_squared_error(y_test,pred_lr))





visualizer = PredictionError(lr)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show();   



plt.figure(figsize=(12,6))

visualizer = ResidualsPlot(lr)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
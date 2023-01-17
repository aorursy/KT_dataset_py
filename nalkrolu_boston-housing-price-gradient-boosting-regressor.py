from yellowbrick.regressor import PredictionError

from yellowbrick.regressor import ResidualsPlot

from yellowbrick.regressor import CooksDistance
from sklearn.datasets import load_boston

from sklearn.tree  import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression,Ridge,Lasso

from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor,AdaBoostRegressor,BaggingRegressor,RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost
from pylab import rcParams

rcParams['figure.figsize'] = 14,6

plt.style.use('seaborn-talk')

#plt.style.available
boston = pd.DataFrame(load_boston().data,columns=load_boston().feature_names)

boston["DESCR"] = load_boston().target

boston.head()
boston.info()
boston.describe()
plt.figure(figsize=(14,6))

sns.heatmap(boston.corr(),annot=True)

plt.show()
plt.figure(figsize=(14,6))

sns.scatterplot(x="LSTAT",y="RM",data=boston,hue="DESCR",palette="coolwarm")

plt.title("LSTAT vs RM")

plt.show()
plt.figure(figsize=(14,6))

sns.countplot(x="RAD",data=boston,palette="prism")

plt.title("Countplot for RAD")

plt.show()
plt.figure(figsize=(14,8))

plt.subplot(2,2,1)

sns.boxplot(y="DIS",x="CHAS",data=boston,palette="inferno")



plt.subplot(2,2,2)

sns.boxplot(y="NOX",x="CHAS",data=boston,palette="inferno")



plt.subplot(2,2,(3,4))

sns.scatterplot(x="DIS",y="NOX",data=boston,palette="inferno",hue="CHAS")

plt.title("Scatterplot for DIS vs NOX")

plt.tight_layout()

plt.show()
plt.figure(figsize=(14,6))

plt.hist(boston["CRIM"],alpha=0.5,density=True,bins=30)

plt.title("Histogram for CRIM")

plt.show()
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

plt.hist(boston.loc[boston["CHAS"]==1]["AGE"],alpha=0.5,density=True,label="CHAS:1")

plt.hist(boston.loc[boston["CHAS"]==0]["AGE"],alpha=0.6,density=True,label="CHAS:0")

plt.title("Histogram for AGE")

plt.legend(loc="best")



plt.subplot(1,2,2)

plt.hist(boston.loc[boston["CHAS"]==1]["TAX"],alpha=0.5,density=True,label="CHAS:1")

plt.hist(boston.loc[boston["CHAS"]==0]["TAX"],alpha=0.6,density=True,label="CHAS:0")

plt.title("Histogram for TAX")

plt.legend(loc="best")

plt.show()
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

plt.hist(boston.loc[boston["CHAS"]==1]["RM"],alpha=0.5,density=True,label="CHAS:1")

plt.hist(boston.loc[boston["CHAS"]==0]["RM"],alpha=0.6,density=True,label="CHAS:0")

plt.title("Histogram for RM")

plt.legend(loc="best")



plt.subplot(1,2,2)

plt.hist(boston.loc[boston["CHAS"]==1]["PTRATIO"],alpha=0.5,density=True,label="CHAS:1")

plt.hist(boston.loc[boston["CHAS"]==0]["PTRATIO"],alpha=0.6,density=True,label="CHAS:0")

plt.title("Histogram for PTRATIO")

plt.legend(loc="best")

plt.show()
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

plt.hist(boston.loc[boston["CHAS"]==1]["INDUS"],alpha=0.5,density=True,label="CHAS:1")

plt.hist(boston.loc[boston["CHAS"]==0]["INDUS"],alpha=0.6,density=True,label="CHAS:0")

plt.title("Histogram for INDUS")

plt.legend(loc="best")



plt.subplot(1,2,2)

plt.hist(boston.loc[boston["CHAS"]==1]["B"],alpha=0.5,density=True,label="CHAS:1")

plt.hist(boston.loc[boston["CHAS"]==0]["B"],alpha=0.6,density=True,label="CHAS:0")

plt.title("Histogram for B")

plt.legend(loc="best")

plt.show()
X,y = load_boston(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.35,random_state=0)
visualizer = CooksDistance()

visualizer.fit(X, y)

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



visualizer = ResidualsPlot(lr)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
net = MLPRegressor(hidden_layer_sizes=(100,),max_iter=800,activation="relu",alpha=0.0192,random_state=42)

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



visualizer = ResidualsPlot(net)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
dt = DecisionTreeRegressor()

dt.fit(X_train,y_train)

print("=====DecisionTreeRegressor=====")

print("Score Test:",dt.score(X_test,y_test))

pred_dt = dt.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred_dt))

print("MSE:",mean_squared_error(y_test,pred_dt))



visualizer = PredictionError(dt)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show();  



visualizer = ResidualsPlot(dt)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
gb = GradientBoostingRegressor()

gb.fit(X_train,y_train)

print("=====GradientBoostingRegressor=====")

print("Score Test:",gb.score(X_test,y_test))

pred_gb = gb.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred_gb))

print("MSE:",mean_squared_error(y_test,pred_gb))



visualizer = PredictionError(gb)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show(); 



visualizer = ResidualsPlot(gb)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
et = ExtraTreesRegressor(n_estimators=50)

et.fit(X_train,y_train)

print("=====ExtraTreesRegressor=====")

print("Score Test:",et.score(X_test,y_test))

pred_et = et.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred_et))

print("MSE:",mean_squared_error(y_test,pred_et))



visualizer = PredictionError(et)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show(); 



visualizer = ResidualsPlot(et)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
ab = AdaBoostRegressor()

ab.fit(X_train,y_train)

print("=====AdaBoostRegressor=====")

print("Score Test:",ab.score(X_test,y_test))

pred_ab = ab.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred_ab))

print("MSE:",mean_squared_error(y_test,pred_ab))



visualizer = PredictionError(ab)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show(); 



visualizer = ResidualsPlot(ab)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
br = BaggingRegressor()

br.fit(X_train,y_train)

print("=====BaggingRegressor=====")

print("Score Test:",br.score(X_test,y_test))

pred_br = br.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred_br))

print("MSE:",mean_squared_error(y_test,pred_br))



visualizer = PredictionError(br)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show(); 



visualizer = ResidualsPlot(br)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
rf = RandomForestRegressor(n_estimators=100)

rf.fit(X_train,y_train)

print("=====RandomForestRegressor=====")

print("Score Test:",rf.score(X_test,y_test))

pred_rf = rf.predict(X_test)

print("MAE:",mean_absolute_error(y_test,pred_rf))

print("MSE:",mean_squared_error(y_test,pred_rf))



visualizer = PredictionError(rf)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show(); 



visualizer = ResidualsPlot(rf)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
xg = xgboost.XGBRFRegressor().fit(X_train,y_train)

print("=====XGBRFRegressor=====")

print("Score Test:",xg.score(X_test,y_test))

pred_xg = xg.predict(X_test)

print("R2 :",r2_score(y_test,pred_xg))

print("MAE:",mean_absolute_error(y_test,pred_xg))

print("MSE:",mean_squared_error(y_test,pred_xg))



visualizer = PredictionError(xg)

visualizer.fit(X_train, y_train)  

visualizer.score(X_test, y_test)        

visualizer.show();     



visualizer = ResidualsPlot(xg)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer

visualizer.score(X_test, y_test)  # Evaluate the model on the test data

visualizer.show();
mse = {

    "LinearRegression":[mean_squared_error(y_test,pred_lr)],

    "MLPRegressor":[mean_squared_error(y_test,pred_net)],

    "DecisionTreeRegressor":[mean_squared_error(y_test,pred_dt)],

    "GradientBoostingRegressor":[mean_squared_error(y_test,pred_gb)],

    "ExtraTreesRegressor":[mean_squared_error(y_test,pred_et)],

    "AdaBoostRegressor":[mean_squared_error(y_test,pred_ab)],

    "BaggingRegressor":[mean_squared_error(y_test,pred_br)],

    "RandomForestRegressor":[mean_squared_error(y_test,pred_rf)],

    "XGBRFRegressor":[mean_squared_error(y_test,pred_xg)]

      }

mse = pd.DataFrame(mse).T

mse.columns=["Value"]



plt.figure(figsize=(14,6))

sns.barplot(y=mse.sort_values("Value").index,x="Value",palette="coolwarm",data=mse.sort_values("Value"))

plt.title("MSE")

plt.draw()
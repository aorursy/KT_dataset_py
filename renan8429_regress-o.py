import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')
Train = pd.read_csv("../input/train.csv")
Test =  pd.read_csv("../input/test.csv")
print(Train.shape)
Train.head()
YTrain = Train['median_house_value']
XTrain = Train.drop(['Id', 'median_house_value'], axis=1)
Test = Test.drop(['Id'], axis=1)

plt.matshow(Train.corr())
Train.corr().iloc[1:9,-1 ].plot('bar')
plt.show()

pd.plotting.scatter_matrix(Train)
plt.show()
y=Train['median_house_value']
x=Train['median_income']
plt.scatter(x,y)
plt.show()
y=Train['households']
x=Train['total_bedrooms']
plt.scatter(x,y)
plt.show()
y=Train['median_age']
x=Train['longitude']
plt.scatter(x,y)
plt.show()
razao= Train['total_rooms']
razao=razao/Train['households']
plt.scatter(razao,Train['median_house_value'] )
plt.show()
razao= Train['median_income']
razao=razao/Train['households']
plt.scatter(razao,Train['median_house_value'] )
plt.show()
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(XTrain, YTrain)
pred= reg.predict(XTrain)
from sklearn.model_selection import cross_val_score
score= cross_val_score(reg,XTrain,YTrain,cv=10)
score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
Score_MSE = MSE(YTrain, pred, multioutput='uniform_average')
Score_MSE
Score_R2= r2_score(YTrain,pred)
Score_R2
def rmsle(y_pred, y_test) : 
    assert len(y_test) == len(y_pred)
    return np.sqrt(np.mean((np.log(1+y_pred) - np.log(1+y_test))**2))
from sklearn.metrics import make_scorer
score_rmsle = make_scorer(rmsle)
Score_rmsle = cross_val_score(reg,XTrain,YTrain,cv=10, scoring=score_rmsle)
Score_rmsle.mean()
X2Train = XTrain.drop(['longitude', 'latitude'], axis=1)
reg2= LinearRegression()
reg2.fit(X2Train, YTrain)
pred2= reg2.predict(X2Train)

score2 = cross_val_score(reg2,X2Train,YTrain,cv=10,scoring=score_rmsle)
score2.mean()
X3Train = XTrain.drop(['median_age'], axis=1)
reg3= LinearRegression()
reg3.fit(X3Train, YTrain)
pred3= reg3.predict(X3Train)
score3= cross_val_score(reg3,X3Train,YTrain,cv=10,scoring=score_rmsle)
score3.mean()


from sklearn.linear_model import Lasso
lasso=Lasso(alpha=5)
lasso.fit(XTrain, YTrain)
score_lasso=cross_val_score(lasso,XTrain,YTrain,cv=10,scoring=score_rmsle)
score_lasso.mean()
from sklearn.linear_model import Ridge
ridge = Ridge(0.9)
ridge.fit(XTrain,YTrain)
score_ridge = cross_val_score(ridge, XTrain, YTrain, cv=10, scoring= score_rmsle)
score_ridge.mean()
    
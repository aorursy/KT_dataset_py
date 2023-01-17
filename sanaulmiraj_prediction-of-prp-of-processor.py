from sklearn import datasets
import pandas as pd

df=pd.read_csv('../input/cpudata.csv')
print(df.head())


#Checking for missing values
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df)
plt.show()

#correalation 

corr=df.corr()
print(corr)
X=df.iloc[:,0:6]
y=df.iloc[:,6]
X=X.as_matrix()
y=y.as_matrix()
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.svm import SVR
import numpy as np
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor

kf = KFold(n_splits=5,random_state = 33,shuffle = True)
#First try a linear model
#reg=linear_model.SGDRegressor(loss='squared_loss',penalty=None, random_state=42)
reg=linear_model.LinearRegression()



accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       scaler = preprocessing.StandardScaler().fit(X_train)
       X_train = scaler.transform(X_train)
       X_test = scaler.transform(X_test)
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print("Avearage r2 score  for linear regression: ",np.mean(accuracy))




    
       

#suppor vector regression
reg=SVR(kernel='linear')
accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       scaler = preprocessing.StandardScaler().fit(X_train)
       X_train = scaler.transform(X_train)
       X_test = scaler.transform(X_test)
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print("Avearage r2 score  for sv regression: ",np.mean(accuracy))
reg= DecisionTreeRegressor(max_depth=5)
accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       scaler = preprocessing.StandardScaler().fit(X_train)
       X_train = scaler.transform(X_train)
       X_test = scaler.transform(X_test)
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print("Avearage r2 score  for dt regression: ",np.mean(accuracy))
reg = RandomForestRegressor(max_depth=5, random_state=0)
accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       scaler = preprocessing.StandardScaler().fit(X_train)
       X_train = scaler.transform(X_train)
       X_test = scaler.transform(X_test)
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print("Avearage r2 score  for rf regression: ",np.mean(accuracy))
print(accuracy)
reg=ExtraTreesRegressor(max_depth=5, random_state=0)
accuracy=[]

for train_index, test_index in kf.split(X):
       
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       
       scaler = preprocessing.StandardScaler().fit(X_train)
       X_train = scaler.transform(X_train)
       X_test = scaler.transform(X_test)
        
       reg.fit(X_train, y_train)
       y_pred = reg.predict(X_test)
       accuracy.append(metrics.r2_score(y_test, y_pred))

print ("Avearage r2 score  for extra tree regression: ",np.mean(accuracy))
import matplotlib.pyplot as plt
plt.plot(y_test,color='green')
plt.plot(y_pred,color='red')
plt.legend(['Actual','Prediction'])
plt.show()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit(X_train).transform(X_train)
X_test=sc.transform(X_test)

import tensorflow as tf
import numpy as np
feature_columns=tf.contrib.learn.infer_real_valued_columns_from_input(X_train)
dnn_clf=tf.contrib.learn.DNNRegressor(hidden_units=[30,30],activation_fn=tf.nn.relu,feature_columns=feature_columns)
dnn_clf.fit(x=X_train,y=y_train,batch_size=20,steps=5000)

print(dnn_clf.evaluate(X_test,y_test))
y_pred=dnn_clf.predict(X_test)
y_prediction=[]
for i in y_pred:
  y_prediction.append(i)
import numpy as np
y_pred=np.array(y_prediction)
print(y_pred.shape)
print(y_test.shape)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
score

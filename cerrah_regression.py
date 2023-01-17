import numpy as np
import pandas as pd 
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df_original = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
df = df_original.dropna()
df.head()
#We are not going to try to fill NaNs instead we cleared the data and looked what we have to do
#we are going to use that a lot so we did a func for it
def Evaluationmatrix(test, predict):
    print ('Mean Squared Error: '+ str(metrics.mean_squared_error(test,predict)))
    print ('Mean absolute Error: '+ str(metrics.mean_absolute_error(test,predict)))
    print ('Mean squared Log Error: '+ str(metrics.mean_squared_log_error(test,predict)))

df.drop(labels = ['Content Rating','Last Updated','Current Ver','Android Ver','App','Genres','Category','Price'], axis = 1, inplace = True)
#We could use Price, Genres and Category too but we prefered droping them
#making installs integers
df['Installs'] = [int(i[:-1].replace(',','')) for i in df['Installs']]
#to make it understandable for program making strings to integers
def cat(cat):
    if cat=="Free":
        return 0
    else:
        return 1

df['Type'] = df['Type'].map(cat)
#changin mb and kb to integers and making them more stable
def size_changer(size):
    if "M" in size:
        X = size[:-1]
        X = float(X)*1000000
        return(X)
    
    elif "k" in size:
        X = size[:-1]
        X = float(X)*1000
        return(X)
    else:
        return None


    
df["Size"] = df["Size"].map(size_changer)
df.Size.fillna(method = 'ffill', inplace = True)
df2 = df
X = df2.drop("Rating",axis=1)
y = df2["Rating"]
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)
lr = LinearRegression()
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
Evaluationmatrix(y_test,pred)
alpha = [0.001,0.1,1]

for new_alpha in alpha:
    r = Ridge(alpha=new_alpha)
    r.fit(X_train,y_train)
    pred2 = r.predict(X_test)
    

Evaluationmatrix(y_test,pred2)
for new_alpha in alpha:
    lasso = Lasso(alpha=new_alpha,max_iter=10e5)
    lasso.fit(X_train,y_train)
    pred3=lasso.predict(X_test)

Evaluationmatrix(y_test,pred3)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
param = {"n_neighbors":np.arange(1,200)}
knr=KNeighborsRegressor()
gs= GridSearchCV(knr,param,cv=10)
gs.fit(X_train,y_train)
pred4=gs.predict(X_test)
Evaluationmatrix(y_test,pred4)
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, y_train)
pred5=regr.predict(X_test)
Evaluationmatrix(y_test,pred5)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
params = {"n_estimators":np.arange(10,200,10)}

gs_rfr = GridSearchCV(rfr,params,cv=10)
gs_rfr.fit(X_train,y_train)
pred6 = gs_rfr.predict(X_test)
Evaluationmatrix(y_test,pred6)
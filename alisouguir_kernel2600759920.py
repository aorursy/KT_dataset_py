import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv("../input/intelligence/intelligenceArti.csv")
data
data.info()
data.describe()
plt.figure(figsize=(20,10))
plt.scatter(data.iloc[:,0],data.iloc[:,1],label= 'Base de données initial')
plt.legend(fontsize=20)
ax = plt.axes()
plt.grid(which='major', axis='x', color='y', linewidth=1)
plt.grid(which='major', axis='y', color='b', linestyle='dashed')
plt.suptitle('Force critique de voilement', fontsize=30)
plt.xlabel('w',fontsize=25)
plt.ylabel('Fcr',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
from sklearn.preprocessing import RobustScaler
from sklearn import preprocessing
x_array = np.array(data['ww'])
y_array = np.array(data['ff'])
l1=preprocessing.normalize([x_array])
data.iloc[:,0]=l1[0]
data=data.sample(frac=1)
data=data.reset_index(drop=True)
X=data.iloc[:,0]
y=data.iloc[:,1]
X
plt.figure(figsize=(20,10))
plt.scatter(X,y,label= 'Base de données normalisé',color="red")
plt.legend(fontsize=20)
ax = plt.axes()
plt.grid(which='major', axis='x', color='y', linewidth=1)
plt.grid(which='major', axis='y', color='b', linestyle='dashed')
plt.suptitle('Force critique de voilement',fontsize=25)
plt.xlabel('w',fontsize=25)
plt.ylabel('Fcr',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print('Train set:', X_train.shape)
print('Test set:', X_test.shape)
plt.figure(figsize=(20, 10))
plt.subplot(121)
plt.scatter(X_train, y_train, alpha=0.8)
plt.title('Train set',fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.subplot(122)
plt.scatter(X_test,y_test,color="red", alpha=0.8)
plt.title('test set',fontsize=20)
plt.suptitle('Division de la base des données en X_train,Y_train,X_test,y_test',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

import xgboost as xgb
regressor = xgb.XGBRegressor(
    n_estimators=50,
    reg_lambda=1,
    gamma=0,
    max_depth=3
)
y_train.reset_index(drop=True)
X.shape
X_train.shape
y_train.shape
regressor.fit(np.array(X_train).reshape(350,1),np.array(y_train).reshape(350,1))
y_pred = regressor.predict(np.array(X_test).reshape(88,1))
print(y_pred)
def calcul(input_r,input_t,start_l,end_l,num_pts):
    input_L=np.linspace(start = start_l, stop = end_l, num = num_pts)
    w=input_L/np.sqrt(input_r*input_t)
    k=np.array(w)
    y_pred = regressor.predict(preprocessing.normalize([np.array(k)]).reshape(num_pts,1))
    return k,y_pred
input_w,output_force=calcul(120,1.2,100,1720,10)
print(input_w)
print(output_force)
plt.figure(figsize=(20,10))
plt.scatter(input_w,output_force,marker="o",label="Fcr with A.I",color="green")
plt.legend( fontsize=20)
ax = plt.axes()
plt.suptitle('Force critique de voilement scattered with Artificial Intelligence', fontsize=30)
plt.grid(which='major', axis='x', color='y', linewidth=1)
plt.grid(which='major', axis='y', color='b', linestyle='dashed')
plt.xlabel('w',fontsize=25)

plt.xlabel('w',fontsize=25)
plt.ylabel('Fcr',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.figure(figsize=(20,10))
plt.plot(input_w,output_force,"gp--",label="Fcr with A.I",linewidth=1.5)
plt.legend( fontsize=20)
ax = plt.axes()
plt.suptitle('Force critique de voilement with Artificial Intelligence', fontsize=30)
plt.grid(which='major', axis='x', color='r', linewidth=1)
plt.grid(which='major', axis='y', color='b', linestyle='dashed')
plt.xlabel('w',fontsize=25)
plt.ylabel('Fcr',fontsize=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


d = {'w':input_w, 'f':output_force}
df = pd.DataFrame(data=d)
df.to_csv("alisouguir.csv")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import category_encoders as ce
dataset = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
dataset.head(10)
y_data=dataset.iloc[:,[2]]
y=y_data.values

#removing cng cars
y=np.delete(y,[18,36],0)

#visual
y_data.head(10)
x_data = dataset.iloc[:,1:]

#encoding categorical variables

ohc_5 = ce.OneHotEncoder(cols=['Transmission','Seller_Type','Fuel_Type'])
x_data=ohc_5.fit_transform(x_data);
x_orig=x_data.values

#removing records with 'Fuel_Type' == 'CNG'
x_orig=np.delete(x_orig,[18,35],0)

#avoiding dummy variable trap
x_orig=np.delete(x_orig,[1,5,6,8,10],1)

#visual
x_data.head(10)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=sc_x.fit_transform(x_orig)
[x_row,x_col]=x.shape
x=np.append(arr=np.ones((x_row,1)).astype(float),values=x,axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
def cost(x,y,theta):
    [m,n]=x.shape
    h=np.dot(x,theta);
    J=(1/(2*m))*np.sum((h-y)**2)
    return J
def gradient_descent(x,y,theta,iterations,alpha):
    [m,n]=x.shape
    theta_opt=theta
    J_history=np.zeros((iterations,1))
    grad_history=np.zeros((iterations,n))
    grad_history=grad_history.reshape(iterations,n)
    
    for i in range(iterations):
        h=np.dot(x,theta_opt);
        grad = (alpha/m)*(np.sum(((h-y)*x),axis=0))
        grad=grad.reshape(-1,1)
        theta_opt= theta_opt - grad
        J_history[i]=cost(x,y,theta_opt)
        grad_history[i]=theta_opt.transpose()
        
    return theta_opt,grad_history,J_history   
theta=np.random.randn(x_col+1,1)
iterations=1000

theta_opt,grad_history,J_history = gradient_descent(x_train,y_train,theta,iterations,0.01)
iter_plt=np.array([range(iterations)]).transpose()
plt.plot(iter_plt,J_history)
plt.xlabel('iterations')
plt.ylabel('cost')
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred_sk=lr.predict(x_test)
y_pred_sk=y_pred_sk.reshape(-1,1)
cost_SK=(1/(2*len(y)))*np.sum((y_pred_sk-y_test)**2)
cost_GD = cost(x_test,y_test,theta_opt)
print("the mean squared error of the sckikit learn model is ",cost_SK)
print("and the mean squared error of the cost of the model that was made manually is ",cost_GD)
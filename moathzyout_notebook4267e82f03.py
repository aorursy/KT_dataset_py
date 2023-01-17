# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
from  numpy import  *
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        data=pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data.head()
import matplotlib.pyplot as plt 
plt.scatter(data['fixed acidity'],data['quality'])
plt.scatter(data['alcohol'],data['quality'])
def pred_y(data,m,b):

    return  m * data + b
pred_y(data['alcohol'],1,1)
featuers_train=[]
for fe in data.drop(['quality'],axis=1).head(0):
    print(fe)
    featuers_train.append(fe)

from sklearn.model_selection import cross_val_score
from numpy import asarray
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def pca_Data(data):
        x = StandardScaler().fit_transform(data)
        pca = PCA(n_components=data.shape[1])
        principalComponents = pca.fit_transform(x)
        principalDf = pd.DataFrame(data = principalComponents
                     , columns =featuers_train )
        return principalDf

X_S=pca_Data(data.drop(['quality'],axis=1))

import plotly.express as px
fig = px.scatter_matrix(X_S)
fig.show()

y=data['quality']

mode_reg=LinearRegression()
Mse=cross_val_score(mode_reg,X_S,y,scoring='neg_mean_squared_error',cv=1000)
print('score',Mse)
Mse_average=np.mean(Mse)
Mse_average
plt.plot(range(len(Mse)),Mse)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(X_S, y)
alpha=ridge_regressor.best_params_['alpha']
ridge = Ridge(alpha=alpha)
ridge.fit(X_S,y)
ridge_mse = cross_val_score(ridge, X_S,y,scoring='neg_mean_squared_error', cv=5)


ridge_mse
plt.plot(ridge_mse)
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 20)

lasso_regressor.fit(X_S, y)
lasso_regressor.best_params_['alpha']

lasso = Lasso(alpha=lasso_regressor.best_params_['alpha'])
lasso_regressor_Mse = cross_val_score(lasso, X_S, y,scoring='neg_mean_squared_error', cv = 50)

lasso_regressor_Mse
plt.plot(lasso_regressor_Mse)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.arange(10,1000,10),"solver":["liblinear"], "penalty":["l2"],"max_iter":np.arange(1000,2500,50)}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_S,y)
scoers=cross_val_score(logreg,X_S,y,cv=10)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)
plt.plot(scoers)
print(scoers.mean())
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
mse=mean_squared_error(y,logreg_cv.predict(X_S))
mse
data['quality'].unique()
loss = []
for label in data['quality'].unique():

    loss.append(log_loss(y.values==label,logreg_cv.predict(X_S)))
    
    
np.mean(loss)
y_pred=logreg_cv.predict(X_S)
y_pred
error=y-y_pred
True_negtive=[]
True_postive=[]
for err in error:
    
    if err==0:
        True_negtive.append(err)
        
    else:
         True_postive.append(err)
TN=len(True_negtive)/len(y)
Tp=len(True_postive)/len(y)
accuercy=((TN+Tp)/len(y))*100
accuercy
Set=set([2,4,5,5,4,6,6])
type(Set)
j=0
groups=data.groupby('quality')
g13=groups.get_group(3)
g14=groups.get_group(4)
g15=groups.get_group(5)
g16=groups.get_group(6)
g17=groups.get_group(7)
g18=groups.get_group(8)



group_dict={"g3":g13,"g4":g14,"g5":g15,"g6":g16,"g7":g17,"g8":g18}
data.head(0)
import plotly.express as px

fig = px.scatter_matrix(group_dict["g3"].iloc[0:,1:5])
fig.show()
import plotly.express as px

fig = px.scatter_matrix(group_dict["g3"].iloc[0:,1:5])
fig.show()
data.head(0)
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
  
  
# scatter plot with regression  
# line(by default) 
sns.lmplot(x ='volatile acidity', y ='pH', data = group_dict["g3"]) 
  
# Show the plot 
plt.show() 
sns.lmplot(x ='volatile acidity', y ='pH', data = group_dict["g4"]) 
  
# Show the plot 
plt.show() 
sns.lmplot(x ='volatile acidity', y ='pH', data = group_dict["g5"]) 
  
# Show the plot 
plt.show() 
sns.lmplot(x ='volatile acidity', y ='pH', data = group_dict["g6"]) 
  
# Show the plot 
plt.show() 
sns.lmplot(x ='volatile acidity', y ='pH', data = group_dict["g7"]) 
  
# Show the plot 
plt.show() 
sns.lmplot(x ='volatile acidity', y ='pH', data = group_dict["g8"]) 
  
# Show the plot 
plt.show() 
sns.lmplot(x ='volatile acidity', y ='pH', 
           fit_reg = False, data = group_dict["g3"]) 
  
# Show the plot 
plt.show() 
sns.lmplot(x ='volatile acidity', y ='pH', 
           fit_reg = False, data = group_dict["g4"]) 
sns.lmplot(x ='volatile acidity', y ='pH', 
           fit_reg = False, data = group_dict["g5"]) 
sns.lmplot(x ='volatile acidity', y ='pH', 
           fit_reg = False, data = group_dict["g6"]) 
sns.lmplot(x ='volatile acidity', y ='pH', 
           fit_reg = False, data = group_dict["g7"]) 
I=np.eye(len(data['pH']))
I.shape

x=data['pH'].values
x.reshape(-1,1)
def fun_square(x):
    x=x.reshape(-1,1)
    m=np.dot(I,x)
    x_s=x-np.std(m)
    return np.square(x_s)
m=fun_square(data['pH'].values)
plt.scatter(data['pH'],m)
d1=data.head(0)
for d in d1 :
    m=fun_square(data[d].values)
    plt.scatter(data[d],m)
    plt.show()
    
    
new_data=pd.DataFrame()
for d in d1:
    d_n=d+'S'
    new_data[d_n]=fun_square(data[d].values).ravel()
    
new_data
F_data=new_data.drop(['qualityS'],axis=1)
F_data

full_data=pd.concat([data,F_data],axis=1)
full_data
y=full_data['quality']
X_S=pca_Data(data.drop(['quality'],axis=1))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
grid={"C":np.arange(10,1000,10),"solver":["liblinear"], "penalty":["l2"],"max_iter":np.arange(1000,2500,50)}# l1 lasso l2 ridge
logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(X_S,y)
scoers=cross_val_score(logreg,X_S,y,cv=10)
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


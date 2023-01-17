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
import seaborn as sns
from sklearn.model_selection import train_test_split
data=pd.read_csv('/kaggle/input/fish-market/Fish.csv')
data.head()
data.info()
data.nunique()
data['Weight'].plot()
from scipy.stats import skew
data['Weight'].skew()
cor_mat=data.corr().round(2)
plt.figure(figsize=(10,7))
sns.heatmap(data=cor_mat,annot=True)
X=data.drop(columns=['Length2','Length3','Weight','Species','Width'])
Y=data['Weight']
import seaborn as sns
sns.boxplot(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

pw=PowerTransformer()
ms=MinMaxScaler()

X_train=ms.fit_transform(X_train)
X_test=ms.transform(X_test)

Y_train=Y_train.values.reshape(-1,1)
Y_test=Y_test.values.reshape(-1,1)

Y_train=pw.fit_transform(Y_train)
Y_test=pw.transform(Y_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

model_lin=LinearRegression()
model_lin.fit(X_train,Y_train)
y_test_predictv=model_lin.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predictv))

r2=r2_score(y_test_predictv,Y_test)

print('the values predicted has')
print('RMSE = {}'.format(rmse))
print('r2 Score= {}'.format(r2))

from sklearn.linear_model import ElasticNet

e_model=ElasticNet(alpha=0.01)

e_model.fit(X_train,Y_train)

y_test_predict=e_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))

r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))
from sklearn.model_selection import GridSearchCV

e_estimator=ElasticNet()

parameters={'alpha':[0.001,0.1,0.3,0.5,0.8,10,11,12],
          'l1_ratio':[0.01,0.1,0.3,0.4,0.8,1]}
grid=GridSearchCV(estimator=e_estimator,param_grid=parameters,cv=2,n_jobs=-1)
grid.fit(X_train,Y_train)
grid.best_params_
from sklearn.linear_model import ElasticNet

e_model=ElasticNet(alpha=0.001,l1_ratio=0.1)

e_model.fit(X_train,Y_train)

y_test_predict=e_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))

r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))
y_test_predict=y_test_predict.reshape(-1,1)
Y_test = pw.inverse_transform(Y_test)
y_test_predict = pw.inverse_transform(y_test_predict)
from matplotlib.pyplot import plot
plot(y_test_predict, label='Pred')
plot(Y_test, label='Actual')
plt.legend(loc='best')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
#Species
plt.figure(figsize=(20,5))

ax=sns.countplot(data['Species'])
for p in ax.patches:
    h=p.get_height()
    w=p.get_width()/2
    ax.text(p.get_x()+w,h+1,
    '{:1}'.format(h),
    ha='center')
plt.show    
#height
plt.figure(figsize=(30,7))

ax=sns.countplot(data['Height'])
for p in ax.patches:
    h=p.get_height()
    w=p.get_width()/2
    ax.text(p.get_x()+w,h,
    '{:1}'.format(h),
    ha='center')
plt.xticks(rotation=90)    
plt.show    
#length
plt.figure(figsize=(30,7))

ax=sns.countplot(data['Length1'])
for p in ax.patches:
    h=p.get_height()
    w=p.get_width()/2
    ax.text(p.get_x()+w,h,
    '{:1}'.format(h),
    ha='center')
plt.xticks(rotation=90)    
plt.show    
#a = data.groupby(['Species'],as_index=False)[['Height']].count() #indexing
#a
a=data.groupby(['Species'])[['Weight']].count()
a
#using plotly
colors = ['cyan']*9
trace1 = go.Bar(
y=a.Weight,
x=a.index,
marker_color=colors
)

df=[trace1]
layout=go.Layout(
    title='species count',
                font=dict(size=16),
                legend=dict(font=dict(size=6)))
figure=go.Figure(data=df,layout=layout)
py.iplot(figure,filename='barchart')

#mean weight
c=data.groupby(['Species'])[['Weight']].mean()
c
plt.figure(figsize=(10,5))
ax = sns.barplot(x=c.index, y='Weight', data=c)
for p in ax.patches:
    h = p.get_height()
    w = p.get_width()/2
    ax.text(p.get_x()+w, h+3,
            '{:.2f}'.format(h),
           ha="center")
plt.show()
#using plotly
colors = ['lightpink','lightblue','orange','cyan','violet','red','lightgreen']
trace1 = go.Bar(
y=c.Weight,
x=c.index,
marker_color=colors,
    marker=dict(color='darkblue')
)

df=[trace1]
layout=go.Layout(
    title='average weight of species',
                font=dict(size=16),
                legend=dict(font=dict(size=6)))
figure=go.Figure(data=df,layout=layout)
py.iplot(figure,filename='barchart')

data['Length']=(data['Length1']+data['Length2']+data['Length3'])/3
plt.plot(data['Length1'], label='1')
plt.plot(data['Length2'], label='2')
plt.plot(data['Length3'], label='3')
plt.plot(data['Length'], label='mean')
plt.legend()
#Min length
data.drop(columns=['Length1','Length2','Length3'],inplace=True)
a = data.groupby(['Species'],as_index=False)[['Length']].min()
a = a.rename(columns={'Length':'Min_length'})
a
data = pd.merge(data, a , on='Species')
#Max weight
a = data.groupby(['Species'],as_index=False)[['Width']].max()
a = a.rename(columns={'Width':'Max_width'})
a

data = pd.merge(data, a , on='Species')
import seaborn as sns
plt.figure(figsize=(10,7))
cor_mat= data.corr()
sns.heatmap(data=cor_mat,annot=True)
# mean height
a = data.groupby(['Species'],as_index=False)[['Height']].max()
a = a.rename(columns={'Height':'Mean_Height'})
a

data = pd.merge(data, a , on='Species')
data['Volume'] =  data['Height'] * data['Width'] * data['Length']
data.drop(columns=['Height','Width','Length'], inplace=True)
import seaborn as sns
plt.figure(figsize=(10,7))
cor_mat= data.corr()
sns.heatmap(data=cor_mat,annot=True)
dumm=pd.get_dummies(data['Species'])
dumm
#adding the above data back to data
data=pd.concat([dumm,data],axis=1)
data
#now creating a spearman model for the data above
import seaborn as sns
plt.figure(figsize=(10,7))
cor_mat= data.corr(method='spearman')
sns.heatmap(cor_mat ,annot=True)
#now here we can see that this is too much of data of seeing every correlation is a tedious work
import seaborn as sns
plt.figure(figsize=(10,7))
cor_mat= data.corr(method='spearman')
sns.heatmap(cor_mat>0.7 ,annot=True)
#here we can see we have bream and perch are correlated to other parameters other than weight
#dropping species as it is categorical data
data.drop(columns=['Species','Perch','Bream'], inplace =True)
data
X=data.drop(columns=['Weight'])
Y=data['Weight']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import MinMaxScaler

pw=PowerTransformer()
ms=MinMaxScaler()

X_train=ms.fit_transform(X_train)
X_test=ms.transform(X_test)

Y_train=Y_train.values.reshape(-1,1)
Y_test=Y_test.values.reshape(-1,1)

Y_train=pw.fit_transform(Y_train)
Y_test=pw.transform(Y_test)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

model_lin=LinearRegression()
model_lin.fit(X_train,Y_train)
y_test_predictv=model_lin.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predictv))

r2=r2_score(y_test_predictv,Y_test)

print('the values predicted has')
print('RMSE = {}'.format(rmse))
print('r2 Score= {}'.format(r2))
from sklearn.linear_model import Ridge

R_model=Ridge(alpha=0.1)
R_model.fit(X_train,Y_train)

y_test_predict=R_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))
r2=r2_score(Y_test,y_test_predict)

print('RMSE={}'.format(rmse))
print('r2 score={}'.format(r2))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
r_estimator=ElasticNet()

parameters={'alpha':[0.001,0.01,0.05,0.1,0.3,0.5,0.8,10,11,12],
           'l1_ratio' : [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
grid=GridSearchCV(estimator=r_estimator,param_grid=parameters,cv=7,n_jobs=11)
grid.fit(X_train,Y_train)
grid.best_params_
e_model=ElasticNet(alpha=0.001,l1_ratio=0.9)
e_model.fit(X_train,Y_train)
y_test_predictv=e_model.predict(X_test)

rmse=np.sqrt(mean_squared_error(Y_test,y_test_predictv))

r2=r2_score(Y_test,y_test_predictv)

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))
plt.plot(Y_test)
plt.plot(y_test_predictv)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=7)
from sklearn.model_selection import KFold
err=[]
y_pred=[]

X_test = ms.transform(X_test)
fold=KFold(n_splits=7)
for train_index, test_index in fold.split(X_train,Y_train):
    x_train, x_test = X_train.iloc[train_index], X_train.iloc[test_index]
    y_train, y_test = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    
    x_train = ms.fit_transform(x_train)
    x_test = ms.transform(x_test)
    # Y_train = pd.DataFrame(Y_train)  
    y_train = y_train.values.reshape(-1,1)
    y_test = y_test.values.reshape(-1,1)
    y_train = pw.fit_transform(y_train)
    y_test = pw.transform(y_test)

    
    
    
    m1 = ElasticNet(alpha=0.001,
                           l1_ratio=0.9)
    m1.fit(x_train,y_train)
    preds = m1.predict(x_test)

    print("err: ",np.sqrt(mean_squared_error(y_test,preds)))
    print("r2square: ",r2_score(y_test,preds))
    err.append(np.sqrt(mean_squared_error(y_test,preds)))
    test_pred = m1.predict(X_test)
    test_pred = test_pred.reshape(-1,1)
    test_pred = pw.inverse_transform(test_pred)
    y_pred.append(test_pred)
np.mean(err)
Y_test = Y_test.reset_index(drop=True)

plt.plot(Y_test)
plt.plot(y_pred[1])
len(y_pred)
for i in  range(0,len(y_pred)):
# y_pred = np.mean(y_pred, 0)
    rmse=np.sqrt(mean_squared_error(Y_test,y_pred[i]))

    r2=r2_score(Y_test,y_pred[i])

    print('RMSE= {}'.format(rmse))

    print('r2 score= {}'.format(r2))
plt.plot(Y_test)
plt.plot(y_pred[6])
y_predm = np.mean(y_pred, 0)
rmse=np.sqrt(mean_squared_error(Y_test,y_predm))

r2=r2_score(Y_test,y_predm)

print('RMSE= {}'.format(rmse))

print('r2 score= {}'.format(r2))
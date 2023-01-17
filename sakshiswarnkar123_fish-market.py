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
df=pd.read_csv('/kaggle/input/fish-market/Fish.csv')

df.head()
df.describe()
df.info()
df.isnull().sum()
df['Weight'].plot()
import matplotlib.pyplot as plt

X=df['Length1']

Y=df['Length2']

plt.plot(X,Y)

plt.show()

import numpy as np

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

x=df['Weight']

num_bins=5

#n,bins,patches=plt.hist(x,num_bins,facecolor='blue',alpha=0.5)

#plt.plot(x,kind='bar')

x.plot.hist(bins=5)

plt.show()
from scipy.stats import skew

df['Weight'].skew()
corr_fish=df.corr().round(2)
import seaborn as sns

plt.figure(figsize=(10,7))

sns.heatmap(data=corr_fish,annot=True)
X=df.drop(columns=['Length2','Length3','Weight','Width','Species'])

Y=df['Weight']
sns.boxplot(Y)
from sklearn.model_selection import train_test_split

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
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

pipe = make_pipeline(StandardScaler(), LinearRegression())

pipe.fit(X_train,Y_train)

#el=StandardScaler()

#X_train=el.fit_transform(X_train)

#X_test=el.transform(X_test)

y_test_predictv=pipe.predict(X_test)



rmse=np.sqrt(mean_squared_error(y_test_predictv,Y_test))



r2=r2_score(y_test_predictv,Y_test)



print('the predicted values has')

print('RMSE={}'.format(rmse))

print('r2 score={}'.format(r2))
from sklearn.linear_model import Lasso

lasso_model=Lasso(alpha=0.1)

lasso_model.fit(X_train,Y_train)
y_test_predict=lasso_model.predict(X_test)

rmse=(np.sqrt(mean_squared_error(Y_test, y_test_predict)))

r2 = r2_score(Y_test, y_test_predict)

print('rmse is {}'.format(rmse))

print('r2_score is {}'.format(r2))
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso_reg=Lasso()

parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],

              "fit_intercept": [True, False],

             }

grid=GridSearchCV(estimator=lasso_reg,param_grid=parameters,cv=2,n_jobs=-1)

grid.fit(X_train,Y_train)
grid.best_params_
grid.best_score_
L_model = Lasso(alpha=0.001)

L_model.fit(X_train, Y_train)



y_test_predict = L_model.predict(X_test)



rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))



r2 = r2_score(Y_test, y_test_predict)



print("The model performance for testing set")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))
from sklearn.linear_model import Ridge

R_model=Ridge()

R_model.fit(X_train, Y_train)
y_test_predict=R_model.predict(X_test)

rmse=(np.sqrt(mean_squared_error(Y_test,y_test_predict)))

r2=r2_score(Y_test,y_test_predict)

print('rmse is {}'.format(rmse))

print('r2_score is {}'.format(r2))
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

lasso_reg = Ridge()

parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],

              "fit_intercept": [True, False],

             }

grid = GridSearchCV(estimator=lasso_reg, param_grid = parameters, cv = 2, n_jobs=-1)

grid.fit(X_train, Y_train)
grid.best_params_
R_model=Ridge()

R_model.fit(X_train,Y_train)



y_test_predict = R_model.predict(X_test)



rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))



r2 = r2_score(Y_test, y_test_predict)



print("The model performance for testing set")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))
from sklearn.linear_model import ElasticNet

E_model = ElasticNet()

E_model.fit(X_train, Y_train)



y_test_predict = E_model.predict(X_test)



rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))



r2 = r2_score(Y_test, y_test_predict)



print("The model performance")

print('RMSE is {}'.format(rmse))

print('R2 score is {}'.format(r2))
from sklearn.linear_model import ElasticNet

from sklearn.model_selection import GridSearchCV

lasso_reg = ElasticNet()

parameters = {"alpha": [0.001,0.01,0.1,0.3,0.5,0.8,1,4,9,10,30],

              "l1_ratio":[0.1,0.3,0.5,0.8],

              "fit_intercept": [True, False],

             }

grid = GridSearchCV(estimator=lasso_reg, param_grid = parameters, cv = 7, n_jobs=-1)

grid.fit(X_train, Y_train)
grid.best_params_
elastica_model = ElasticNet(alpha=0.001,

                           l1_ratio=0.8)
elastica_model.fit(X_train,Y_train)



y_test_predict=elastica_model.predict(X_test)



rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))



r2=r2_score(Y_test,y_test_predict)



print('RMSE={}'.format(rmse))



print('r2 score={}'.format(r2))
Y_test = pw.inverse_transform(Y_test)

y_test_predict = pw.inverse_transform(y_test_predictv)

from matplotlib.pyplot import plot

plot(y_test_predict, label='Pred')

plot(Y_test, label='Actual')

plt.legend(loc='best')
import matplotlib.pyplot as plt

x=df['Species']

y=df['Length1']

plt.figure(figsize=(10,7))

plt.xlabel('Species')

plt.ylabel('Length1')

plt.legend()

plt.plot(x,y)

plt.show()
import matplotlib.pyplot as plt

x=df['Species']

y=df['Length1']

plt.figure(figsize=(10,7))

plt.xlabel('Species')

plt.ylabel('Length1')

plt.legend()

plt.scatter(x,y)

plt.show()
import matplotlib.pyplot as plt

x=df['Species']

y=df['Length1']

plt.figure(figsize=(10,7))

plt.xlabel('Species')

plt.ylabel('Length1')

plt.legend()

plt.bar(x,y)

plt.show()
import matplotlib.pyplot as plt



data = {'Bream': 10, 'Roach': 15, 'Whitefish': 5, 'Pike': 20}                                 #categorical plotting

names = list(data.keys())

values = list(data.values())



fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

axs[0].bar(names, values)

axs[1].scatter(names, values)

axs[2].plot(names, values)

fig.suptitle('Categorical Plotting')
#comparision of length1 and length3 that which one is better

plt.hist(df['Length1'],facecolor='indigo',edgecolor='black',bins=5,alpha=0.5)

plt.hist(df['Length3'],facecolor='grey',edgecolor='blue',bins=5,alpha=0.3)

plt.show()
import matplotlib.pyplot as plt



x=df['Length1']

y=df['Length2']

z=df['Length3']



fig, axs = plt.subplots(1, 3, figsize=(20, 5), sharey=True)

axs[0].plot(x)

axs[1].plot(y)

axs[2].plot(z)

fig.suptitle('Plottings')
plt.figure(figsize=(10,5))

ax=sns.countplot(df['Species'])

for p in ax.patches:

    h=p.get_height()

    w=p.get_width()/2

    ax.text(p.get_x()+w,h+1,

           '{:1}'.format(h),

           ha="center")

plt.show()
a=df.groupby(['Species'])[['Weight']].count()

a
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
colors = ['cyan',]  *7

trace1 = go.Bar(

    y=a.Weight,

    x=a.index,

    marker_color=colors,

    name='Men',

)



dataa = [trace1]

layout = go.Layout(

    title='Count of Number of Species',

    font=dict(

        size=16

    ),

    legend=dict(

        font=dict(

            size=6

        )

    )

)

fig = go.Figure(data=dataa, layout=layout)

py.iplot(fig, filename='barchart')
b=df.groupby(['Species'])[['Weight']].mean()

b
plt.figure(figsize=(10,5))

ax = sns.barplot(x=a.index, y='Weight', data=a)

for p in ax.patches:

    h = p.get_height()

    w = p.get_width()/2

    ax.text(p.get_x()+w, h+1,

            '{:.2f}'.format(h),

           ha="center")

plt.show()
colors = ['cyan','red','green','lightpink','blue','orange'] 

trace1 = go.Bar(

    y=a.Weight,

    x=a.index,

    marker_color=colors,

    name='Men',

    marker=dict(

        color='darkblue'

    )

)



dataa = [trace1]

layout = go.Layout(

    title='Average Weight of Species',

    font=dict(

        size=16

    ),

    legend=dict(

        font=dict(

            size=30

        )

    )

)

fig = go.Figure(data=dataa, layout=layout)

py.iplot(fig, filename='barchart')
df['Length']=(df['Length1']+df['Length2']+df['Length3'])/3
plt.plot(df['Length1'], label='1')

plt.plot(df['Length2'], label='2')

plt.plot(df['Length3'], label='3')

plt.plot(df['Length'], label='F')

plt.legend()
df.drop(columns=['Length1','Length2','Length3'], inplace=True)
a = df.groupby(['Species'],as_index=False)[['Length']].min()

a = a.rename(columns={'Length':'min_length'})

a
df = pd.merge(df, a , on='Species')
import seaborn as sns

plt.figure(figsize=(10,7))

cor_mat= df.corr()

sns.heatmap(data=cor_mat,annot=True)
a = df.groupby(['Species'],as_index=False)[['Width']].max()

a = a.rename(columns={'Width':'Max_width'})

a
df=pd.merge(df,a,on='Species')
import seaborn as sns

plt.figure(figsize=(10,7))

cor_mat= df.corr()

sns.heatmap(data=cor_mat,annot=True)
df['Volume'] =  df['Height'] * df['Width'] * df['Length']
import seaborn as sns

plt.figure(figsize=(10,7))

cor_mat= df.corr()

sns.heatmap(data=cor_mat,annot=True)
df.drop(columns=['Height','Width','Length'], inplace=True)
import seaborn as sns

plt.figure(figsize=(10,7))

cor_mat= df.corr()

sns.heatmap(data=cor_mat,annot=True)
df
df['Species'].value_counts()
import seaborn as sns

plt.figure(figsize=(10,7))

cor_mat= df.corr(method='spearman')

sns.heatmap(cor_mat,annot=True)
result=pd.get_dummies(df['Species'])
result
df
df=pd.concat([result,df],axis=1)
df
import seaborn as sns

plt.figure(figsize=(10,7))

cor_mat= df.corr(method='spearman')

sns.heatmap(cor_mat ,annot=True)
df.drop(columns=['Species', 'Bream','Perch'], inplace=True)
df
X = df.drop(columns=['Weight'])

Y = df['Weight']
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
elastica_model = ElasticNet(alpha=0.001,

                           l1_ratio=0.9

                          )

elastica_model.fit(X_train,Y_train)



y_test_predict=elastica_model.predict(X_test)



rmse=np.sqrt(mean_squared_error(Y_test,y_test_predict))



r2=r2_score(Y_test,y_test_predict)



print('RMSE={}'.format(rmse))



print('r2 score={}'.format(r2))
plt.plot(Y_test)

plt.plot(y_test_predict)
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
plt.plot(Y_test)

plt.plot(y_pred[2])
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
plt.plot(Y_test)

plt.plot(y_predm)
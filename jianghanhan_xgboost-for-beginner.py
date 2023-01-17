# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://upload.wikimedia.org/wikipedia/commons/6/69/XGBoost_logo.png")



from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "https://i.ytimg.com/vi/dMulLZKm_pg/maxresdefault.jpg")





from xgboost import XGBRegressor 

from sklearn.ensemble import RandomForestRegressor 

from sklearn.linear_model import LinearRegression 

from sklearn.datasets import load_boston

from sklearn.model_selection import KFold,cross_val_score,train_test_split

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt 

from time import time 

import datetime 
##导入数据 

data = load_boston()



x = data.data  #特征 

y = data.target  #目标
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3 ,random_state = 400)



reg = XGBRegressor(n_estimators = 100).fit(x_train,y_train)



reg.score(x_test,y_test)



y_pre = reg.predict(x_test) 



mean_squared_error(y_test,y_pre)

#查看模型重要分数

reg.feature_importances_
#XGBRegressor

reg = XGBRegressor(n_estimators = 100)

print(cross_val_score(reg,x_train,y_train,cv = 5).mean())  ##交叉验证取平均值

print(cross_val_score(reg,x_train,y_train,cv = 5,scoring = 'neg_mean_squared_error').mean()) #交叉验证举平均值 用不同方法

#RandomForestRegressor

rfr = RandomForestRegressor(n_estimators=100)

print(cross_val_score(rfr,x_train,y_train,cv = 5).mean())

print(cross_val_score(rfr,x_train,y_train,cv = 5,scoring = 'neg_mean_squared_error').mean())
Lr = LinearRegression()

print(cross_val_score(Lr,x_train,y_train,cv = 5).mean())

print(cross_val_score(Lr,x_train,y_train,cv = 5,scoring = 'neg_mean_squared_error').mean())
##开启参数查看

reg = XGBRegressor(n_estimators = 100,silent = False)

cross_val_score(reg,x_train,y_train,cv = 5,scoring = 'neg_mean_squared_error').mean()





def plot_learning_curve(estimator,title,x,y

                       ,ax = None      #子图

                       ,ylim = None    #y的范围

                       ,cv = None      

                       ,n_jobs = None  #设定所要使用的线程

                       ):

    from sklearn.model_selection import learning_curve

    import matplotlib.pyplot as plt

    import numpy as np

    

    train_sizes,train_scores,test_scores = learning_curve(estimator,x,y

                                                         ,shuffle=True

                                                         ,cv=cv

                                                         ,random_state=400

                                                         ,n_jobs = n_jobs)

    if ax == None:

        ax = plt.gca()

    else :

        ax = plt.figure()

    ax.set_title(title)

    if ylim is not None:

        ax.set_ylim(*ylim)

    ax.set_xlabel('Training examples')

    ax.set_ylabel('score')

    ax.grid()

    ax.plot(train_sizes,np.mean(train_scores,axis = 1),'o-'

    ,color = 'r',label = 'Training score')

    ax.plot(train_sizes,np.mean(test_scores,axis = 1),'o-'

    ,color = 'g',label = 'Test score')

    ax.legend(loc = 'best')

    return ax 

    
cv = KFold(n_splits = 5,shuffle = True,random_state = 400)

plot_learning_curve(XGBRegressor(n_estimators = 100,random_state = 400)

                   ,'XGBRegressor',x_train,y_train,ax=None,cv=cv)

plt.show()
axisx = range(10,1010,50)

rs = []

for i in axisx :

    reg = XGBRegressor(n_estimators = i , random_state = 400)

    rs.append(cross_val_score(reg,x_train,y_train,cv=cv).mean())

print(axisx[rs.index(max(rs))],max(rs))

plt.figure(figsize = (20,5))

plt.plot(axisx,rs,c = 'red',label ="XGBRegressor")

plt.legend()

plt.show()



axisx = range(50,1050,10) 

rs = [] 

var = [] 

ge = [] 

for i in axisx:

    reg = XGBRegressor(n_estimators=i,random_state=400)

    cvresult = cross_val_score(reg,x_train,y_train,cv=cv)

    

    #记录1-偏差      

    

    rs.append(cvresult.mean())

    #记录方差      

    var.append(cvresult.var())

   

    #计算泛化误差的可控部分        

    ge.append((1 - cvresult.mean())**2+cvresult.var()) 

    

#打印R2高所对应的参数取值，并打印这个参数下的方差     

print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))]) 

#打印方差低时对应的参数取值，并打印这个参数下的R2 

print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var)) 

#打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分 

print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge)) 





plt.figure(figsize=(20,5)) 

plt.plot(axisx,rs,c="red",label="XGBRegressor") 

plt.legend() 

plt.show()

axisx = range(0,90,2)

rs = []

var = [] 

ge = []

for i in axisx:

    reg = XGBRegressor(n_estimators=i,random_state=400)

    

    cvresult = cross_val_score(reg,x_train,y_train,cv=cv)

    rs.append(cvresult.mean())

    var.append(cvresult.var())

    ge.append((1 - cvresult.mean())**2+cvresult.var()) 

print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))

rs = np.array(rs)

var = np.array(var)*0.01

plt.figure(figsize=(20,5))

plt.plot(axisx,rs,c="black",label="XGB") 



#添加方差线 包围起来!! 

plt.plot(axisx,rs+var,c="red",linestyle='-.')

plt.plot(axisx,rs-var,c="red",linestyle='-.')



plt.legend()

plt.show()

#看看泛化误差的可控部分如何？ 

plt.figure(figsize=(20,5))

plt.plot(axisx,ge,c="gray",linestyle='-.')

plt.show()
##检测模型效果



time0 = time()

print(XGBRegressor(n_estimators=60,random_state=400).fit(x_train,y_train).score(x_test,y_test))

print(time()-time0)

time0 = time()

print(XGBRegressor(n_estimators=6,random_state=400).fit(x_train,y_train).score(x_test,y_test))

print(time()-time0)

time0 = time()

print(XGBRegressor(n_estimators=32,random_state=400).fit(x_train,y_train).score(x_test,y_test))

print(time()-time0)
#重要参数subsample



axisx = np.linspace(0,1,20)

rs = []

for i in axisx:

    reg = XGBRegressor(n_estimators=32,subsample=i,random_state=400)

    rs.append(cross_val_score(reg,x_train,y_train,cv=cv).mean())

print(axisx[rs.index(max(rs))],max(rs))

plt.figure(figsize=(20,5))

plt.plot(axisx,rs,c="green",label="XGBRegressor")

plt.legend()

plt.show()

#细化学习曲线

axisx = np.linspace(0.05,1,20)

rs = []

var = []

ge = []

for i in axisx:

    reg = XGBRegressor(n_estimators=180,subsample=i,random_state=400)

    cvresult = cross_val_score(reg,x_train,y_train,cv=cv)

    rs.append(cvresult.mean())

    var.append(cvresult.var())

    ge.append((1 - cvresult.mean())**2+cvresult.var())

    

print(axisx[rs.index(max(rs))],max(rs),var[rs.index(max(rs))])

print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))

print(axisx[ge.index(min(ge))],rs[ge.index(min(ge))],var[ge.index(min(ge))],min(ge))

rs = np.array(rs)

var = np.array(var)

plt.figure(figsize=(20,5)) 



plt.plot(axisx,rs,c="black",label="XGB")

plt.plot(axisx,rs+var,c="red",linestyle='-.')

plt.plot(axisx,rs-var,c="red",linestyle='-.')

plt.legend()

plt.show()

#继续细化学习曲线 

axisx = np.linspace(0.75,1,25)

#不要盲目找寻泛化误差可控部分的低值，注意观察结果

#看看泛化误差的情况如何

reg = XGBRegressor(n_estimators=180,subsample=0.7708333333333334 ,random_state=420).fit(x_train,y_train)

reg.score(Xtest,Ytest)

mean_squared_error(y_test,reg.predict(x_test))
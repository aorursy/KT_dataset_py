import pandas as pd

import numpy as np
video = pd.read_csv('../input/vgsales.csv')

video.head()
video.info()
video.isnull().sum()
video.Name.unique()
print(video[video['Publisher'].isnull()].shape[0])

video[video['Publisher'].isnull()]
video[video['Name'].str.contains('Cartoon Network')][['Name','Publisher','Genre','Platform']]
video['Publisher'].fillna('Unknown',inplace=True)
video.loc[[2236,8368],'Publisher'] = 'Konami Digital Entertainment'
video.loc[8896,'Publisher'] = 'Majesco Entertainment'
video.loc[6849,'Publisher'] = 'THQ'
video.loc[15788,'Publisher'] = 'Activision'
video.loc[14698,'Publisher'] = 'Rondomedia'
video.loc[12517,'Publisher'] = 'Tecmo Koei'
video.loc[8162,'Publisher'] = 'THQ'
video.loc[13278,'Publisher'] = 'Capcom'
video.loc[12487,'Publisher'] = 'Konami Digital Entertainment'
video.loc[15915,'Publisher'] = 'Zoo Games'
video.loc[16198,'Publisher'] = 'Namco Bandai Games'
video.loc[16229,'Publisher'] = 'Ubisoft'
video.loc[6562,'Publisher'] = 'Take-Two Interactive'
video.loc[1303,'Publisher'] = 'Electronic Arts'
video.loc[[4145,6437],'Publisher'] = 'Sega'
video.loc[6272,'Publisher'] = 'Nintendo'
video.loc[8503,'Publisher'] = 'Take-Two Interactive'
video.loc[16191,'Publisher'] = 'Vivendi Games'
video.loc[14942,'Publisher'] = 'Alchemist'
video.loc[16494,'Publisher'] = 'D3Publisher'
video.loc[7953,'Publisher'] = 'Unknown'
video.loc[15261,'Publisher'] = 'Nintendo'
video.loc[[3166,3766,7470],'Publisher'] = 'THQ'
video.loc[[8330,5302],'Publisher'] = 'Atari'
video.loc[470,'Publisher'] = 'THQ'
video.loc[8848,'Publisher'] = 'Nintendo'
video.loc[16553,'Publisher'] = 'Focus Home Interactive'
video.loc[1662,'Publisher'] = 'Activision'
video.loc[8341,'Publisher'] = 'Global Star'
video.loc[14296,'Publisher'] = 'Namco Bandai Games'
video.loc[9749,'Publisher'] = 'Namco Bandai Games'
video.loc[16208,'Publisher'] = 'Banpresto'
video.loc[10494,'Publisher'] = 'Konami Digital Entertainment'
video.loc[10382,'Publisher'] = 'Disney Interactive Studios'
video.loc[[15325,7208,6042,3159],'Publisher'] = 'THQ'
video.loc[[4526,4635],'Publisher'] = 'THQ'
video.loc[9517,'Publisher'] = 'Focus Home Interactive'
video.loc[7351,'Publisher'] = 'Atari'
video.loc[13962,'Publisher'] = 'Wargaming.net'
video.corr()
dummydata = pd.get_dummies(video.drop(['Year','Name'],axis=1))

dummydata.head()
from sklearn.preprocessing import StandardScaler
scaleddata = pd.DataFrame(StandardScaler().fit_transform(dummydata),columns=dummydata.columns)

scaleddata.head()
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import r2_score,mean_squared_error
x = scaleddata.drop('Global_Sales',axis=1)

y = scaleddata['Global_Sales']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state=0,test_size=0.30)
lr = LinearRegression()
lr.fit(xtrain,ytrain)
ypred = lr.predict(xtest)
r2_score(ytest,ypred)
np.sqrt(mean_squared_error(ytest,ypred))
r = Ridge(alpha=0.5,random_state=0,solver='lsqr')
r.fit(xtrain,ytrain)
ypredR = r.predict(xtest)

ypredR
r2_score(ytest,ypredR)
np.sqrt(mean_squared_error(ytest,ypredR))
l = Lasso()
l.fit(xtrain,ytrain)

ypredLasso = l.predict(xtest)

ypredLasso
r2_score(ytest,ypredLasso)
np.sqrt(mean_squared_error(ytest,ypredLasso))
enet = ElasticNet()
enet.fit(xtrain,ytrain)
ypredEnet = enet.predict(xtest)

ypredEnet
r2_score(ytest,ypredEnet)
np.sqrt(mean_squared_error(ytest,ypredEnet))
params = {"alpha":[0.01,0.5,1,2,3,4,0.02,0.03,0.09,10,50],

         "solver":["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],

         "random_state":[0,2,1,3,500]}

params_lasso = {"alpha":[0.01,0.5,1,2,3,0.05,0.02,0.03,0.09,0.001,5],

         "random_state":[0,2,1,3,500]}

params_elastic = {"alpha":[0.01,0.5,1,2,3,0.05,0.02,0.03,0.001,5],

         "random_state":[0,2,1,3,500]}

grid = GridSearchCV(estimator=r,param_grid=params,cv =3)
#grid.fit(xtrain,ytrain)
#grid.best_params_
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score

d=pd.read_csv('../input/fifa19/data.csv')

d=d.loc[d['Position']=='GK']

gk=d[['Age','Overall','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes']]

gk=gk.dropna()

X=gk[['Age','GKDiving','GKHandling','GKKicking','GKPositioning','GKReflexes']]

y=gk[['Overall']]

lr=LinearRegression()

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.33,random_state=5)

lr.fit(Xtrain,ytrain)

predte=lr.predict(Xtest)

#sns.regplot(Xtest['Age'],ytest,color='black')

#sns.regplot(Xtest['GKHandling'],ytest,color='yellow')

#sns.regplot(Xtest['GKDiving'],ytest,color='red')

r2=r2_score(ytest,predte)

mse=mean_squared_error(ytest,predte)

print (r2)

print (mse)

plt.scatter(predte,ytest)

#plt.plot(ytest,predte,linewidth=3)









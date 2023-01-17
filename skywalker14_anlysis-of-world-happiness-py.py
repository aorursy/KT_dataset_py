import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

df=pd.read_csv('../input/2015.csv')

df.head(2)



x=np.array(df['Happiness Score'])

ax=sns.distplot(x)



corr=df.corr()

sns.heatmap(corr,vmax=0.8,square=True)
y=df['Happiness Score']

var1=df['Family']

plt.scatter(var1,y)

plt.show()
y=df['Happiness Score']

var2=df['Freedom']

plt.scatter(var2,y)

plt.show()
y=df['Happiness Score']

var3=df['Dystopia Residual']

plt.scatter(var3,y)

plt.show()
X = df.drop(['Happiness Score','Happiness Rank','Country','Region'],axis=1)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

trainx,testx,trainy,testy = train_test_split(X,y,test_size=0.2,random_state=101)

lm=LinearRegression()

lm.fit(trainx,trainy)

predicty=lm.predict(testx)
plt.scatter(testy,predicty)

plt.xlabel('TEST y')

plt.ylabel('Predict y')

plt.show()
coefficients=pd.DataFrame(lm.coef_,X.columns)

coefficients
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(testy, predicty))

print('MSE:', metrics.mean_squared_error(testy, predicty))

print('RMSE:', np.sqrt(metrics.mean_squared_error(testy, predicty)))
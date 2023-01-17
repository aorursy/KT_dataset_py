import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt

from collections import OrderedDict



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score
#loading the data

tr=pd.read_csv('../input/train.csv')
test=pd.read_csv('../input/train.csv')
# the correlation between numerical variables

corr = tr.corr()

mask = np.array(corr)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(20,10)

sns.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")
tr.head()
tr['title'].value_counts()[:50].plot(kind='bar',figsize=(60,20),fontsize=10)
tr.plot.scatter('budget','revenue')
tr_mean = tr.groupby('title').mean().sort_values(by='runtime', ascending=False)[['runtime']]
tr_mean.plot(kind='bar',figsize=(80,20),color='green').set_title('runtime for title')
tr['original_language'].value_counts().plot(kind='bar',figsize=(60,10),fontsize=20).set_title('plotting original languages count')
tr[['title','revenue']].groupby(['title']).sum().plot()
#visualizing original title vs revenue

trg=tr[:20]

trg[['revenue','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='blue').set_title('original title vs revenue for first 20 data',fontsize=40)
#visualizing original title vs revenue least 50 data

trg=tr[-50:]

trg[['revenue','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='blue').set_title('original title vs revenue for least 50 data',fontsize=40)
#visualizing original title vs budget

trg=tr[:20]

trg[['budget','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='pink').set_title('original title vs budget for first 20 data',fontsize=40)
#visualizing original title vs budget least 50

trg=tr[-50:]

trg[['budget','original_title']].groupby(['original_title']).sum().plot(kind='bar',figsize=(50,10), fontsize=40,color='pink').set_title('original title vs budget least 50 data',fontsize=40)
#visualizing original title vs budget least 50

trg=tr[-50:]

trg[['popularity','original_title']].groupby(['original_title']).sum().plot(kind='hist',figsize=(50,10), fontsize=40,color='pink').set_title('original title vs budget least 50 data',fontsize=40)
tr.head()
df=tr[:10]

ta=df.sort_values('revenue',ascending=False)

tit = df['title']

rbr = ta['revenue']

colors  = ("red", "green", "orange", "cyan", "brown", 

"grey","blue","indigo", "beige", "yellow")

plt.pie(rbr, labels=tit, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=90)

plt.title('Top 10 data high revenue titles',fontsize=40)

plt.show()
df=tr[:10]

ta=df.sort_values('revenue',ascending=False)

tit = df['title']

rbr = ta['runtime']

colors  = ("red", "green", "orange", "cyan", "brown", 

"grey","blue","indigo", "beige", "yellow")

plt.pie(rbr, labels=tit, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Top 10 data more runtime titles',fontsize=40)

plt.show()
df=tr[:10]

ta=df.sort_values('budget',ascending=False)

tit = df['title']

rbr = ta['budget']

colors  = ("red", "green", "orange", "cyan", "brown", 

"grey","blue","indigo", "beige", "yellow")

plt.pie(rbr, labels=tit, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Top 10 high budget titles',fontsize=40)

plt.show()
tr.info()
tr.dtypes
tr.head()
tr.shape
# missing values in the datasaet

tr.isnull().sum()
tr.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split
y=tr['revenue']
x_train, x_test, y_train, y_test = train_test_split(tr, y, test_size=0.2,random_state=0)
x_train=tr.drop('revenue',axis=1)
x_test=tr.drop('revenue',axis=1)
y_train=tr['revenue']
y_test=tr['revenue']
# checking the shape of X_train, y_train, X_val and y_val

x_train.shape, y_train.shape, x_test.shape, y_test.shape
#performing linear regression as the target variable is a continuous data i.e.,revenue
from sklearn.linear_model import LinearRegression
x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)
#filling NaN values
x_train.fillna(0,inplace=True)
x_test.fillna(0,inplace=True)
# applying dummies on the train dataset

tr=pd.get_dummies(tr)
lg=LinearRegression()
# fitting the model on X_train and y_train

lg.fit(x_train,y_train)
#post processing predicting values
# making prediction on validation set

pred=lg.predict(x_test)
pred
sol=pd.DataFrame()
sol['predicted values']=pred
pred1=lg.predict(x_train)
pred1
lg.score(x_train,y_train)
lg.score(x_test,y_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,pred,multioutput='raw_values')
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_train,pred1,multioutput='raw_values')
# defining a function which will return the rmsle score

def rmsle(y, y_):

    y = np.exp(y),   # taking the exponential as we took the log of target variable

    y_ = np.exp(y_)

    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))

    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))

    calc = (log1 - log2) ** 2

    return np.sqrt(np.mean(calc))
rmsle(y_test,pred)
rmsle(y_train,pred1)
from sklearn.tree import DecisionTreeClassifier
# defining the decision tree model with depth of 4, you can tune it further to improve the accuracy score

clf = DecisionTreeClassifier(max_depth=4, random_state=0)
# fitting the decision tree model

clf.fit(x_train,y_train)
# making prediction on the validation set

predict = clf.predict(x_test)
predict
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data= pd.read_csv('../input/train.csv')

data.head()
data.info()
data.describe()
from datetime import datetime
data['datetime']=data['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

data['year']=data['datetime'].apply(lambda x:x.year)

data['month']=data['datetime'].apply(lambda x:x.month)

data['day']=data['datetime'].apply(lambda x:x.day)

data['hour']=data['datetime'].apply(lambda x:x.hour)

data.drop('datetime',axis=1,inplace=True)

data.head()
data=data[['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',

       'humidity', 'windspeed', 'casual', 'registered',  'year',

       'month', 'day', 'hour','count']]

data.head()
pca_data=data

temp=pca_data.values

np.random.shuffle(temp)

pca_data.columns
temp=pd.DataFrame(temp,columns=pca_data.columns)
temp.head()
from sklearn.preprocessing import scale
temp=scale(temp.values)

temp=pd.DataFrame(temp,columns=pca_data.columns)

temp.head()
import seaborn as sns

sns_plot=sns.pairplot(temp)

#sns_plot.savefig("Datapairplot.png")
temp.corr()
pca_train=temp.drop('count',axis=1)

pca_test=temp['count']
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression

pca=PCA(n_components=2)

pca.fit(pca_train)

xpca=pca.transform(pca_train)
pca_train.shape
xpca.shape
pca.components_
xpca[:]
pca_train=pd.DataFrame(xpca[:],columns=['Component 1','Component 2'])

pca_train.head()
sns.distplot(pca_train['Component 1'])
sns.distplot(pca_train['Component 2'])
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(pca_train, pca_test, test_size=0.38, random_state=42)

linmodel=LinearRegression()

linmodel.fit(X_train,y_train)

linmodel.score(X_test,y_test)
from sklearn.model_selection import cross_val_score

kscore=cross_val_score(linmodel,pca_train,pca_test,scoring='r2',cv=10)
kscore.mean()
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(n_estimators=200,random_state=0)

rf.fit(X_train,y_train)

rf.score(X_test,y_test)
rfkscore=cross_val_score(rf,pca_train,pca_test,scoring='r2',cv=5)
rfkscore.mean()
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

sc_Y=StandardScaler()

X_trtemp=sc_X.fit_transform(X_train)

y_trtemp=sc_Y.fit_transform(y_train.reshape(-1,1))

X_tetemp=sc_X.fit_transform(X_test)

y_tetemp=sc_Y.fit_transform(y_test.reshape(-1,1))





from sklearn.svm import SVR

svr=SVR(kernel='rbf')

svr.fit(X_trtemp,y_trtemp)

svr.score(X_tetemp,y_tetemp)
accuracies=cross_val_score(svr,sc_X.fit_transform(pca_train),sc_Y.fit_transform(pca_test.reshape(-1,1)),cv=10)

print (accuracies.mean())
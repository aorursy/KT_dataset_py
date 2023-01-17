import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from IPython.display import Image

from IPython.core.display import HTML 

Image(url= "http://images.memes.com/meme/592575")
df=pd.read_csv('../input/creditcard.csv')

test=df.copy()
df.shape
df.head()
df.describe()
plt.figure(figsize=(35,20))

plt.plot(df.drop(['Time','Amount','Class'],axis=1))

plt.legend(df.drop(['Time','Amount','Class',],axis=1).columns)

plt.show()
Image(url= "http://s2.quickmeme.com/img/db/dbc97d3b537a3b38f323b2cd9e97228de9342018e72bb18e3b36ec235a8783f5.jpg")
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title('Feature V15')

#ax1.set_yaxis('Value')

ax1.plot(df['V15'])



ax2.set_title('Amount')

#ax2.yaxis('Value')

ax2.plot(df['Amount'])



f.set_figheight(5)

f.set_figwidth(15)

Image(url='https://i.imgflip.com/1htaug.jpg',)
plt.figure(figsize=(20,8))

plt.ylim(0,4000)

plt.plot(df['Amount'])

plt.plot(df[df['Class']==1]['Amount'],'ro')

plt.show()
df['Time'][284806]/(60*60*24)  #The data provided is for almost 2 days 

                               #which is also mentioned in the dataset description -_-
pos=np.where(df['Time'].values >= 60*60*24)

df['Time'].values[pos]=df['Time'].values[pos]-60*60*24
from sklearn.feature_selection import VarianceThreshold

feat=df.iloc[:,0:df.shape[1]-1]



X=feat.values

y=df['Class'].values



sel = VarianceThreshold()

sel.fit_transform(X).shape
from sklearn.tree import ExtraTreeClassifier

from sklearn.ensemble import RandomForestClassifier

plt.figure(figsize=(15,20))

et=ExtraTreeClassifier()

et=et.fit(X,y)

fi=et.feature_importances_

plt.barh(np.arange(len(feat.columns)),fi,align='center',alpha=0.8)

plt.yticks(np.arange(len(feat.columns)),list(feat.columns))

plt.title('Feature importance using Ensemble method',fontsize=20)

plt.ylim(-1,len(feat.columns))

plt.show()
threshold=0.02

sel_features=list(feat.columns[np.where(fi>=threshold)])

sel_features
from sklearn.model_selection import cross_val_score



score=cross_val_score(et,X,y)

score.mean()
X_new=df[sel_features]

et=ExtraTreeClassifier()

et=et.fit(X_new,y)

print(cross_val_score(et,X_new,y).mean())
Image(url='http://denisuca.com/wp-content/uploads/2014/06/thinking-meme-640x523.png')
threshold=0.01

sel_features=list(feat.columns[np.where(fi>=threshold)])

len(sel_features)
X_new=df[sel_features]

from sklearn.tree import DecisionTreeClassifier



dt=DecisionTreeClassifier()

dt=dt.fit(X_new,y)

print(cross_val_score(dt,X_new,y))
Image(url='https://cdn.meme.am/cache/images/folder883/5018883.jpg')
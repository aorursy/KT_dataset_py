# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
data=pd.read_csv('../input/dataset.csv')
data.head()

data.columns

data=data.replace('?',np.nan)
columns=['Account1',  'History', 'Motive','Account2','Employment Period', 'Gender&Type','Sponsors', 'Plotsize', 

         'Plan', 'Housing', 'Post','Phone', 'Expatriate']
data=pd.get_dummies(data, columns=columns)
cols=['Monthly Period','Credit1','InstallmentRate','Tenancy Period','Age', '#Credits', '#Authorities','Class']
data[cols].isnull().sum()
for col in cols:

    print(col)

    data[col] = data[col].fillna(-1)

    data[col] = data[col].astype(int)

    data[col] = data[col].replace(-1, np.nan)
data.info()
colsfl=['InstallmentRate','Yearly Period','InstallmentCredit']

for col in colsfl:

    print(col)

    data[col] = data[col].fillna(-1)

    data[col] = data[col].astype(float)

    data[col] = data[col].replace(-1, np.nan)

    
colsft=['Monthly Period','Credit1','InstallmentRate','Yearly Period','InstallmentCredit','Age','Tenancy Period']

data[colsft]=data[colsft].fillna(data[colsft].mean())

lab=['1','2','3']

for col in colsft:

    print(col)

    data[col]=pd.cut(data[col],3,labels=lab)

    data[col] = data[col].astype(int)
train=data[data['Class'].notna()]

test=data[data['Class'].isnull()]
train=train.fillna(train.mean())
y=train['Class']
train=train.drop(columns=['Class'])
X=train
test=test.drop(columns=['Class'])
test=test.fillna(test.mean())
train.head()
X=X.drop(columns=['id'])
test_id=test['id']
test=test.drop(columns=['id'])
from sklearn.preprocessing import StandardScaler
X.groupby(y).mean()
X=X.drop(columns=['InstallmentCredit','Monthly Period'])

test=test.drop(columns=['InstallmentCredit','Monthly Period'])
scaler=StandardScaler()

scaler.fit(X)



X=scaler.transform(X)

tes=scaler.transform(test)

from sklearn.decomposition import PCA

pca=PCA(n_components=2)

pca.fit(X)

X=pca.transform(X)

test=pca.transform(test)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10,stratify=y)
# from sklearn.neighbors import KNeighborsClassifier

# neigh = KNeighborsClassifier(n_neighbors=3)
# neigh.fit(X_train, y_train) 
# y_pred=neigh.predict(X_test)
from sklearn.metrics import accuracy_score
# accuracy_score(y_test, y_pred)
# from sklearn.tree import DecisionTreeClassifier
# tree=DecisionTreeClassifier()
# tree.fit(X_train,y_train)
# y_tree=tree.predict(X_test)
# accuracy_score(y_test, y_tree)
# from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf.fit(X_train, y_train)
# y_gg=clf.predict(X_test)

# y_gg=y_gg.astype(int)
# accuracy_score(y_test, y_gg)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0)

kmeans.fit(X_train,y_train)
y_mean=kmeans.predict(X_test)

accuracy_score(y_test, y_mean)
# out=clf.predict(test)

# out=out.astype(int)
out=kmeans.predict(test)

out=out.astype(int)
output = pd.DataFrame( { 'id': test_id , 'Class': out } )
output.to_csv( '2015A7Ps0031G.csv' , index = False )
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(output)
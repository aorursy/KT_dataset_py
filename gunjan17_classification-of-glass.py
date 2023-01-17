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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

% matplotlib inline
#now take the data from csv

data=pd.read_csv("../input/glass.csv")
data.describe()
data.head()
#now check the which type of glass is present

type = data.groupby(['Type'])['RI'].count()
type


type.plot(kind='barh')
col = data.columns[:-1].tolist()

X = data[col].values

y=data['Type'].values













X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)

#now the import the classifier

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_train,y_train)
pre=knn.predict(X_test)
from sklearn.metrics import f1_score
ytest = np.array(y_test)

pre = np.array(pre)
f1_score(ytest,pre,average='micro')
# lets see how many entries are really accurate

count = 0

for i in range(0,len(ytest)):

    if ytest[i]==pre[i]:

        count+=1

count
pre


ytest


#so % accuracy is 

acc=count/(len(pre)+1)
acc = acc*100


acc
len(pre)
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
import pandas as  pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
glass = pd.read_csv("../input/glass.csv")
from sklearn.neighbors import KNeighborsClassifier as KNC

from sklearn.model_selection import train_test_split



glass['RI'].value_counts()

glass['Na'].value_counts()

glass['Mg'].value_counts()

glass['Al'].value_counts()

glass['Si'].value_counts()

glass['K'].value_counts()

glass['Ca'].value_counts()

glass['Ba'].value_counts()

glass['Fe'].value_counts()

glass['Type'].value_counts()



np.mean(glass.Mg)

np.max(glass.Mg)-np.min(glass.Mg)

np.min(glass.RI)

glass['Mg']=np.where(glass.Mg > 2.68,'1','0')



glass['Mg'].value_counts()



np.mean(glass.K)

np.max(glass.K)-np.min(glass.K)

np.min(glass.K)

glass['K']=np.where(glass.K > 0.49,'1','0')



glass['K'].value_counts()





np.mean(glass.Ba)

np.max(glass.Ba)-np.min(glass.Ba)

np.min(glass.Ba)

glass['Ba']=np.where(glass.Ba > 0.17,'1','0')



glass['Ba'].value_counts()





np.mean(glass.Fe)

np.max(glass.Fe)-np.min(glass.Fe)

np.min(glass.Fe)

glass['Fe']=np.where(glass.Fe> 0.05,'1','0')



glass['Fe'].value_counts()





train,test=train_test_split(glass,test_size=0.2,random_state=0)



#to get perfect n_neighboors need to perform parameter tuning or bagging method

neigh=KNC(n_neighbors=3)





neigh.fit(train.iloc[:,0:9],train.iloc[:,9])





train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])



test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])



acc=[]

for i in range(1,100,2):

          neigh=KNC(n_neighbors=i)

          neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

          train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])

          test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])

          acc.append([train_acc,test_acc])

          

          

          

          

          

import matplotlib.pyplot as plt

plt.plot(np.arange(1,100,2),[i[0] for i in acc],"-ro")

plt.plot(np.arange(1,100,2),[i[1] for i in acc],"-bo")

plt.legend(["train","test"])

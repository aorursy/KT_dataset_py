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



df = pd.read_csv('../input/fertilizer1/FertPredictDataset.csv')

df2 = pd.read_csv('../input/fertilizer2/data.csv')
df2.head()
df2 = df2.drop(['Phosphorous'] ,axis=1)
df2
X = df2.iloc[:,:3]

y = df2.iloc[:,3].round(2)
X = np.asarray(X)

y = np.asarray(y)
from sklearn.preprocessing import MinMaxScaler
newX = X[:,:2]
newX = newX/1000
X = np.column_stack((newX,X[:,2]))
mm = MinMaxScaler()



X = mm.fit_transform(X)
from sklearn.linear_model import LinearRegression  
lr = LinearRegression()



lr.fit(X,y)
Xnew = df.loc[:,['N','K','C']]
ph = lr.predict(Xnew)
ph = ph.round(2)
ph = np.reshape(ph,(-1,1))
ph
lrnew = LinearRegression()

lrnew.fit(ph,np.asarray(df.loc[:,'class']))
from sklearn.externals import joblib
from sklearn.svm import SVC
svm = SVC()
svm.fit(ph,np.asarray(df.loc[:,'class']))
model  = 'fertilizer2.sav'



joblib.dump(svm,model)
ph[101]
result = svm.predict(ph)
np.asarray(df.loc[:,'class'])
result
for i in result:

    print(i)

import os

for i,j,k in os.walk('.'):

    for filename in k:

        print(filename)
ph
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
df = pd.read_csv('../input/train.csv')

df.head()
df.Top10.head()

Y = df.Top10
from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC
X = df

X.drop('Top10',axis=1,inplace = True)

X.drop('songtitle',axis=1,inplace = True)

X.drop('artistname',axis=1,inplace = True)

X.drop('songID',axis=1,inplace = True)

X.drop('artistID',axis=1,inplace = True)
Lr = LogisticRegression()

Lr.fit(X,Y)
test = pd.read_csv('../input/test.csv')
Xt = test

Xt=Xt.drop('songtitle',axis=1)

Xt=Xt.drop('artistname',axis=1)

Xt=Xt.drop('songID',axis=1)

Xt=Xt.drop('artistID',axis=1)
Ypred = test.loc[:,'songID']
Ypred=pd.DataFrame(Ypred)
Ypred['Top10']=pd.DataFrame(np.zeros(Ypred.shape[0]))
Ypred.head()
H= Lr.predict_proba(Xt)
H=pd.DataFrame(H)
Ypred.Top10 = H[1]
Ypred.to_csv('submission.csv',index=False)
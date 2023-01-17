# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from pandas import Series, DataFrame

#import pandas as pd

#import numpy as np

from sklearn.linear_model import LogisticRegression





X=DataFrame(pd.read_table('../input/train.csv',sep=','))

Y=X[['Survived']]

X.drop(['PassengerId','Name','Ticket','Fare','Cabin','Survived'],axis=1,inplace=True)

#(C = Cherbourg->0; Q = Queenstown->1; S = Southampton->2)

#(Male->0; Female->1)

X.replace(['male','female','C','Q','S'],[0,1,0,1,2],inplace=True)

X.fillna(X.mean(),inplace=True)



Xtest=DataFrame(pd.read_table('../input/test.csv',sep=','))

id=Xtest[['PassengerId']]

Xtest.drop(['PassengerId','Name','Ticket','Fare','Cabin'],axis=1,inplace=True)

Xtest.replace(['male','female','C','Q','S'],[0,1,0,1,2],inplace=True)

Xtest.fillna(Xtest.mean(),inplace=True)





#Ytest=DataFrame(pd.read_table('gender_submission.csv',sep=','))

#Ytest.drop(['PassengerId'],axis=1,inplace=True)





modelLR = LogisticRegression()

modelLR = modelLR.fit(X, np.ravel(Y))

modelLR.score(X, Y)



pLR = modelLR.predict(Xtest)



LR=pd.concat([id,DataFrame(pLR,columns=['Survived'])],axis=1)

LR.to_csv('submission.csv',index=False)





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
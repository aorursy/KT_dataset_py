# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv", na_values="NA")

test = pd.read_csv("../input/test.csv", na_values="NA")



y = train['Survived'].copy()

train.drop(['Survived'],axis=1,inplace=True)

X=pd.concat([train,test])

m=train.shape[0]

n=test.shape[0]
X['Sex']=pd.get_dummies(X['Sex'])

X['Embarked']=X['Embarked'].fillna("S")

p=pd.get_dummies(X['Embarked'])

X=pd.concat([X,p],axis=1)



a=np.zeros((m+n,1))

for i in range(0,m+n+1):

    if(X['Cabin'][i].empty):

        a[i]=0

    else:

        c=1

        for j in range(0,len(X['Cabin'][i])):

            if(X['Cabin'][i][j]==' '):

                c=c+1

        a[i]=c

a







X.drop(['PassengerId','Name','Ticket','Embarked'],axis=1,inplace=True)



x=X.as_matrix()

x = np.nan_to_num(x)



x_train=x[:int(train.shape[0] * 0.8)]

x_cv= x[int(train.shape[0] * 0.8):train.shape[0]]

x_test=x[train.shape[0]:]

y_train=y[:int(train.shape[0] * 0.8)]

y_cv=y[int(train.shape[0]*0.8):]

"""alphas=[0.5,1,100,1000,5000,6000,7000,8000,10000]

errors={}

for alpha in alphas:

    lr=LogisticRegression(penalty='l1',intercept_scaling=alpha)

    lr.fit(x_train,y_train)



    h = lr.predict_proba(x_cv)

    h=h[:,1]

    sq_diff = -y_cv*np.log(h)-(1-y_cv)*np.log(1-h)

    errors[alpha] = (np.sum(sq_diff) /y_cv.shape[0])

print(errors)"""



lr=LogisticRegression(penalty='l1',intercept_scaling=5000)

lr.fit(x_train,y_train)

Y=lr.predict(x_test)

out = pd.DataFrame()

out['PassengerId'] = [i for i in range(train.shape[0]+1,train.shape[0]+x_test.shape[0]+1)]

out['Survived'] = Y

out.to_csv("output.csv", index=False)

print(out)
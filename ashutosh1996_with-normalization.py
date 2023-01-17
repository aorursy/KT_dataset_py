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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression





train=pd.read_csv("../input/train.csv", na_values="NA")

test = pd.read_csv("../input/test.csv", na_values="NA")



y = train['Survived']

train.drop(['Survived'],axis=1,inplace=True)

X=pd.concat([train,test])

X.drop(['PassengerId','Ticket','Cabin','Embarked','Name'],axis=1,inplace=True)

X['Sex']=pd.get_dummies(X['Sex'])

m1=X.loc[:,'Age'].mean()

m2=X.loc[:,'Fare'].mean()

max1=X.loc[:,'Age'].max()

min1=X.loc[:,'Age'].min()

max2=X.loc[:,'Fare'].max(axis=0)

min2=X.loc[:,'Fare'].min(axis=0)

X1=(X.loc[:,'Age']).as_matrix()

X2=(X.loc[:,'Fare']).as_matrix()





for i in range(0,X.shape[0]):

    X1[i]=(X2[i]-m1)/(max1-min1)

    X2[i]=(X2[i]-m2)/(max2-min2)

x=X.as_matrix()



x[:,2]= X1

x[:,5]= X2





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



lr=LogisticRegression(penalty='l1',intercept_scaling=7000)

lr.fit(x_train,y_train)

Y=lr.predict(x_test)

out = pd.DataFrame()

out['PassengerId'] = [i for i in range(train.shape[0]+1,train.shape[0]+x_test.shape[0]+1)]

out['Survived'] = Y

out.to_csv("output2.csv", index=False)

#print(out)

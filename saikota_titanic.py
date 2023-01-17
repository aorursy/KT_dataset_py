# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df=pd.read_csv("../input/train.csv")

test_data=pd.read_csv("../input/test.csv")

print(test_data.columns[0:1])

passenger_mat=(test_data.as_matrix(columns=test_data.columns[0:1]))

print(passenger_mat)

new=df

print(df)

# Any results you write to the current directory are saved as output.
new=new.drop('Name',axis=1)

test_new=test_data.drop('Name',axis=1)

print (new)
new=new.drop('PassengerId',axis=1)

test_new=test_new.drop('PassengerId',axis=1)

print (new)

new=pd.get_dummies(new,columns=['Sex','Embarked'])

test_new=pd.get_dummies(test_new,columns=['Sex','Embarked'])

print (new)
new=new.drop('Ticket',axis=1)

new=new.drop('Cabin',axis=1)

test_new=test_new.drop('Ticket',axis=1)

test_new=test_new.drop('Cabin',axis=1)

print(new)
new=new.fillna(new.mean())

test_new=test_new.fillna(test_new.mean())

print(new)

print(test_new)
mat=new.as_matrix()

test_mat=test_new.as_matrix()

print(mat)

print(test_mat)
output=mat[:,0]

print(output)

inpt=(mat[:,[i for i in range(1,11)]])

print(inpt)
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf=clf.fit(inpt,output)
predicted=clf.predict(test_mat)

predicted=predicted.reshape(predicted.shape[0],-1)

print(predicted)

print(predicted.shape)



print(passenger_mat.shape)

#a=(passenger_mat[:,0])

#print(a.shape)

final=np.concatenate((passenger_mat,predicted),axis=1)

print(final)

df=pd.DataFrame(final)

df=df.astype(int)

print(df)

df.to_csv("./submit.csv",sep=',',header=["PassengerId","Survived"],index=False)
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
train_loc="../input/train.csv"



train_loc2="../input/test.csv"

train_loc3="../input/sample_submission.csv"





df=pd.read_csv(train_loc)



df2=pd.read_csv(train_loc2)



df3=pd.read_csv(train_loc3)

df.shape
X=df.iloc[:,1:]

y=df.iloc[:,0]

X.shape

#disp=np.reshape([X],(28,28))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1)

print(X_train.shape)

print(X_test.shape)
#DECISION TREE CLASSIFIER

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()

classifier.fit(X,y)
y_new=classifier.predict(X_test)
y_new
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_new)
cm
df3
temp=classifier.predict(df2.iloc[:,:])

temp.shape
temp2=np.array([])

for i in range(28000):

    temp2=np.append(temp2,i+1).astype(int)
new_df=pd.DataFrame()
new_df["ImageId"]=temp2

new_df["Label"]=temp
new_df
new_df.to_csv('submissions_new_2.csv', index = False)
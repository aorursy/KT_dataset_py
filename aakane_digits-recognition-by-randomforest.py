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
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#train.shape
#test.shape
#train.head()
#test.head()
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
%matplotlib inline

a = train.iloc[6,1:].values
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)
plt.show()
df_x = train.iloc[:,1:] #feature values
df_y = train.iloc[:,0] #label values
x_train, x_test, y_train, y_test = train_test_split(df_x,df_y,test_size = 0.2,random_state=4)
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
y = y_test.values
count = 0
for i in range(len(pred)):
    if pred[i] == y[i]:
        count += 1

#calculate accuracy of this model 
acc = count/len(pred)
print(acc)
X_test = test.iloc[:,:].values
Y_test = rf.predict(X_test)
a = X_test[77]
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)
plt.show()
Y_test[77]

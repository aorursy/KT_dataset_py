# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.linear_model

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test= pd.read_csv('../input/test.csv')

train=pd.read_csv('../input/test.csv')



import matplotlib.pyplot as plt

plt.scatter(train.x,train.y)
"""linear regression model"""

model = sklearn.linear_model.LinearRegression()



"""training data"""

X_train=np.array(train['x']).reshape(-1,1)

y_train= train['y']



model.fit(X_train,y_train)



"""predicting the target value"""





X_test= np.array(train['x']).reshape(-1,1)

pred_y= model.predict(X_test)



print(pred_y)
"""visualization"""



plt.scatter(X_test, test['y'], color = "g",

               marker = "o", s = 30)



plt.plot(X_test,pred_y, color = "r")



plt.xlabel('x')

plt.ylabel('y')



plt.show()
accuracy = r2_score(pred_y,test['y']) * 100

print('Accuracy of the model :-',accuracy)
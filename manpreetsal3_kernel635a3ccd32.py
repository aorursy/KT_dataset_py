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
#logistic regression

#->classfication problem 

#->earlier we used linear eq y=b0+b1x1+b2x2+....

#->now we will usesigmoid function for classication (y=1/1-e^y) 





import numpy as np

import pandas as pd

from matplotlib import pyplot as plt



#importing the dataset



dataset=pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')



#independent

X=dataset.iloc[:,[2,3]].values

#depenedent

Y=dataset.iloc[:,4].values



#splting the dataset train/test

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=0.25,random_state=0)



#feature scaling

#differnece between the data is too high



from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_transform=sc.fit_transform(xtrain)

xtest=sc.fit_transform(xtest)



#model creation for logistic regression

from sklearn.linear_model import LogisticRegression

lg=LogisticRegression(random_state=0)

lg.fit(xtrain,ytrain)



#predict the test data

y_pred=lg.predict(xtest)



#confusion matrix

from sklearn.metrics import confusion_matrix

cn=confusion_matrix(ytest,y_pred)

cn



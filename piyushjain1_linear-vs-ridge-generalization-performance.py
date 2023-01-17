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

#********************************************************************************************************************************************************************************************************************************

#********************************************************************************************************************************************************************************************************************************

#********************************************************************************************************************************************************************************************************************************

#********************************************************************************************************************************************************************************************************************************

#********************************************************************************************************************************************************************************************************************************

#***********************************************************  This Part is for Linear and  Ridge  vs various split size and alpha value******************************************************************************************





#***************************** Below is for Data import in X and y  *********************************



from sklearn.datasets import load_boston

from sklearn.datasets import load_breast_cancer

Bunchboston=load_boston()

BunchCancer=load_breast_cancer()

# Selecteddata=input('input which data to select\t')

# if Selecteddata=='cancer' : 

X=pd.DataFrame(BunchCancer.data)

y=pd.DataFrame(BunchCancer.target)

# elif Selecteddata=='boston':

#     X=pd.DataFrame(Bunchboston.data)

#     y=pd.DataFrame(Bunchboston.target)

lrTrainScore=[]

lrTestScore=[]

print('selected Data shape is ',X.shape)



#***************************** Below is for Linear  Regresson *********************************

#***************************** Below is for Linear  Regresson *********************************

#***************************** Below is for Linear  Regresson *********************************

splitsize=np.linspace(.05,.6,100)

for i in splitsize:

    from sklearn.model_selection import train_test_split

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=i, random_state=42)

    from sklearn.linear_model import LinearRegression

    lr=LinearRegression()

    lr.fit(X_train,y_train)

    lrTrainScore.append(lr.score(X_train,y_train))

    lrTestScore.append(lr.score(X_test,y_test))

    #print('\n *******selected data is ',Selecteddata)

    #print('Split Size is {:.2f}'.format(i))

    #print('Size of X {0} Size of y {1} // Size of X_train {2}  y_train {3} //'.format(X.shape,y.shape,X_train.shape,y_train.shape,))

    #print('Coefficients are \n {0} \n Intercept is {1} '.format(lr.coef_,lr.intercept_))

    #print('training set score is {:.2f}'.format(lr.score(X_train,y_train)))

    #print('testing set score is {:.2f}'.format(lr.score(X_test,y_test)))



import matplotlib.pyplot as plt

%matplotlib notebook

plt.figure(figsize=(15,8))

#plt.plot(np.linspace(.5,.8,30),np.linspace(.5,.8,30))

plt.xlabel('X- axis along Representing  SplitSize')

plt.ylabel('Y- Axis showing Scores')

plt.plot(splitsize,lrTrainScore,label='Train Score',marker='o')

plt.plot(splitsize,lrTestScore,label='Test Score',marker='s')

plt.title('Linear Regression various splitsize and test score ')

plt.legend()





#***************************** Below is for Ridge Regresson *********************************



#***************************** Below is for Ridge Regresson *********************************



#***************************** Below is for Ridge Regresson *********************************



#print('max of test score is {:.2f} when splitsize is {:.2f}'.format(max(lrTestScore),splitsize))



from sklearn import linear_model

ridgeTrainScore=[]

ridgeTestScore=[]



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.23, random_state=42)





for i in np.linspace(-.1,10,100):

     ridge=linear_model.Ridge(alpha=i)    

     ridge.fit(X_train,y_train)

     ridgeTrainScore.append(ridge.score(X_train,y_train))

     ridgeTestScore.append(ridge.score(X_test,y_test))

    

plt.figure(figsize=(15,8))

#print(splitsize,ridgeTestScore,ridgeTrainScore)

plt.plot(np.linspace(-.1,10,100),ridgeTrainScore,label='train scores',marker='o')

plt.plot(np.linspace(-.1,10,100),ridgeTestScore,label='test scores',marker='s')

plt.xlabel('X-axis Representing alpha values for L2 Regularization')

plt.ylabel('Y- Axis showing Scores')

plt.title('Ridge Alpha Regularization vs Test Score  ')

plt.legend()

plt.show()







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
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split



BunchCancer=load_breast_cancer()

X=BunchCancer.data

y=BunchCancer.target



from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt

#plt.style(



plt.figure(figsize=(30,10),dpi=100)

plt.xticks(rotation=45)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.04,shuffle=False)



for j,marker in zip([.001,.01,.1,1,10,100],['s','o','*','P','d','p']):

    model=LogisticRegression(C=j)

    model.fit(X_train,y_train)

    plt.xticks(rotation=45)

    plt.title('Coeffients and impact of Regularization in Logistic ')

    plt.scatter(BunchCancer.feature_names,model.coef_,marker=marker)

    #plt.plot(BunchCancer.feature_names,lasso.coef_,label= str(i) + 'alpha')



plt.legend(labels=['Alpha .001','Alpha .01','Alpha .1','Alpha 1','Alpha 10','Alpha 100'],loc=0)

plt.show()



#     print('\n printing Lasso Alpha and Training Score as {:.3f}  / {:.2f}'.format(j,lasso.score(X_train,y_train)))

#     print('printing Lasso Alpha and testing Score as {:.3f}  / {:.2f}'.format(j,lasso.score(X_test,y_test)))

#     print('Coeffcients for above for model is {}',lasso.coef_)

    

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split



BunchCancer=load_breast_cancer()

X=BunchCancer.data

y=BunchCancer.target



from sklearn.svm import LinearSVC





import matplotlib.pyplot as plt

#plt.style(



plt.figure(figsize=(30,10),dpi=100)

plt.xticks(rotation=45)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.04,shuffle=False)



for j,marker in zip([.001,.01,.1,1,10,100],['s','o','*','P','d','p']):

    model=LinearSVC(C=j)

    model.fit(X_train,y_train)

    plt.xticks(rotation=45)

    plt.title('Coeffients and impact of Regularization in SVC')

    plt.scatter(BunchCancer.feature_names,model.coef_,marker=marker)



plt.legend(labels=['Alpha .001','Alpha .01','Alpha .1','Alpha 1','Alpha 10','Alpha 100'],loc=0)

plt.show()



#     print('\n printing Lasso Alpha and Training Score as {:.3f}  / {:.2f}'.format(j,lasso.score(X_train,y_train)))

#     print('printing Lasso Alpha and testing Score as {:.3f}  / {:.2f}'.format(j,lasso.score(X_test,y_test)))

#     print('Coeffcients for above for model is {}',lasso.coef_)

    
import sklearn as sklearn 

sklearn.get_config()
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
#read the data

df=pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

df.head(10)
df.info()
#check the no. of classes

df['class'].unique()
#use standard test split function to split data 

from sklearn.model_selection import train_test_split

X=df.loc[:,df.columns!='class']

Y=df.loc[:,'class']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y)
#import classifcation method

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)
#evaluation metrics we use jaccard index should be 1

from sklearn.metrics import jaccard_score

#score for normal and abnormal

Nor,Abnor=jaccard_score(Y_test,y_pred,pos_label='Normal'),jaccard_score(Y_test,y_pred,pos_label='Abnormal')

Nor,Abnor
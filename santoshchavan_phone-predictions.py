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
df = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
print('The shape of the data is', df.shape[0] )
#Check for null values

(df.isnull().sum()/df.count())*1000
#Reading the columns

df.columns
#Extracting the features

X = df.drop(['price_range'],1)
#Extracting the label

y = df['price_range'].copy()
#Splitting the train and test data

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)
#Initiating a SVM model

from sklearn.svm import LinearSVC

model = LinearSVC(random_state = 0)

#Fitiing the model on the train data

model.fit(X_train,y_train)
print('The accuracy of the model is:' ,model.score(X_test,y_test))
#Regularization parameters



c_model_1 = LinearSVC(C=4,random_state = 0)

c_model_1.fit(X_train,y_train)

acc_1 = c_model_1.score(X_test,y_test)





c_model_2 = LinearSVC(C=0.01,random_state = 0)

c_model_2.fit(X_train,y_train)

acc_2 = c_model_2.score(X_test,y_test)





c_model_3 = LinearSVC(C=0.005,random_state = 0)

c_model_3.fit(X_train,y_train)

acc_3 = c_model_3.score(X_test,y_test)

print('The accuracy with C=4 is:', acc_1)

print(50*'==')

print('The accuracy with C=0.01 is:', acc_2)

print(50*'==')

print('The accuracy with C=0.005 is:', acc_3)
#Trying with different types of Kernels

from sklearn.svm import SVC

poly_model = SVC(kernel = 'poly', random_state = 0)

poly_model.fit(X_train,y_train)

acc_poly = poly_model.score(X_test,y_test)

print('The accuracy of Poly:', acc_poly)



print(50*'=')



from sklearn.svm import SVC

rbf_model = SVC(kernel = 'rbf', random_state = 0)

rbf_model.fit(X_train,y_train)

acc_rbf = rbf_model.score(X_test,y_test)

print('The accuracy of rbf:', acc_poly)
#Multi class SVM

# One Vs All

model_ova = SVC(random_state = 0,kernel = 'linear', decision_function_shape = 'ova')

model_ova.fit(X_train,y_train)

acc_ova = model_ova.score(X_test,y_test)

print('Accuracy of One vs all:',acc_ova)



#One vs One



model_ovo = SVC(random_state = 0, kernel = 'linear', decision_function_shape = 'ova')

model_ovo.fit(X_train,y_train)

acc_ovo = model_ovo.score(X_test,y_test)

y_pred = model_ovo.predict(X_test)

print('Accuracy of One vs One:', acc_ovo)

print((y_pred).tolist())
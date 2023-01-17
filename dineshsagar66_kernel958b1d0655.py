##I have created this model in most easiest and understandable way and the pipeline is also very easy

####DINESH SAGAR
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

df=pd.read_csv("/kaggle/input/hr-analytics-people-management/hr_analytics.csv")

df.head()



# Any results you write to the current directory are saved as output.
df.info()
#lets encode the categorical data.

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df.sales=le.fit_transform(df.sales)

df.salary=le.fit_transform(df.salary)

#So for x we need all the data expect LEFT column. for y we need only LEFT



x=df.drop('left',axis=1)

y=df['left']





# lets check whether i selected correct x values or not



x
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x, y , test_size = 0.2, random_state=1) #I have taken 20% test size.
# Using Linear Model and select Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(C=100,solver='lbfgs')

#model = LogisticRegression()

model.fit(X_train,Y_train)
#lets predict



prediction = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix

accuracy_score(prediction,Y_test)
# I was not happy with Logistic regression model, so lets check with support vector.



from sklearn.svm import SVC

svm=SVC(C=100, kernel='rbf', degree=3, gamma='scale')

a=svm.fit(X_train,Y_train)
prediction=svm.predict(X_test)

prediction
from sklearn.metrics import confusion_matrix

print(confusion_matrix(Y_test,prediction))
a=svm.score(X_test,Y_test)

print("The accuracy of the SVC Model is {}".format(a))
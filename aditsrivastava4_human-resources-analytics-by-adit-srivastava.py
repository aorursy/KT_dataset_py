# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





df = pd.read_csv('../input/HR_comma_sep.csv')

#reading the data from the HR_comma_sep.csv file and storing it in df



x = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','salary']]

#getting features from df

features = np.array(x)



y = df[['left']]

lables = np.array(y)

#lables will be if the employee left the company or not



df.head()
features[0]
#changing the salary from string to numbers

#'low' = 1

#'medium' = 2

#'high' = 3



for x in features:

    if(x[6]=='low'):#6 because salary is the 7th element in it

        x[6] = 1

    elif(x[6]=='medium'):

        x[6] = 2

    elif(x[6]=='high'):

        x[6] = 3

    

features[0]
from sklearn.cross_validation import train_test_split

x_train,x_test,y_train,y_test = train_test_split(features,lables,test_size = 0.4, random_state = 42)

len(x_train)
#Support Vector Machine(SVM)



from sklearn.svm import SVC

clf = SVC()

print(clf.fit(x_train,y_train))



pred = clf.predict(x_test)

print(pred)



print("acc = ",clf.score(x_test,y_test))
c = 0

for x in range(0,len(y_test)):

    if(pred[x]==y_test[x]):

        c = c+1



print(len(y_test)," = y_test")

print(c," = c")
from sklearn import metrics

print("Precision score = ",metrics.precision_score(y_test,pred))

print("Recall score = ",metrics.recall_score(y_test,pred))

#Decision Tress Classifire



from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(splitter='random')

print(clf.fit(x_train,y_train))

pred = clf.predict(x_test)

print(pred)



print("acc = ",clf.score(x_test,y_test))
c = 0

for x in range(0,len(y_test)):

    if(pred[x]==y_test[x]):

        c = c+1



print(len(y_test)," = y_test")

print(c," = c")
from sklearn import metrics

print("Precision score = ",metrics.precision_score(y_test,pred))

print("Recall score = ",metrics.recall_score(y_test,pred))

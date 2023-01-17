# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

from sklearn import tree



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_test_lable = pd.read_csv('../input/genderclassmodel.csv')





X = df_train[['Pclass', 'Age', 'Sex',  'Fare']]

y = df_train[['Survived']]

X_train = np.array(X)#train features

y_train = np.array(y)#train lables



X = df_test[['Pclass', 'Age', 'Sex',  'Fare']]

X_test = np.array(X)#test features



y = df_test_lable[['Survived']]

y_test = np.array(y)#test lables



for x in X_train:

    if(x[2]=='male'):

        x[2] = 1

    elif(x[2]=='female'):

        x[2] = 0

    if(str(x[1])=='nan'):

        x[1] = np.random.randint(1,60)



for x in X_test:

    if(x[2]=='male'):

        x[2] = 1

    elif(x[2]=='female'):

        x[2] = 0

    if(str(x[1])=='nan'):

        x[1] = np.random.randint(1,60)

    if(str(x[3])=='nan'):

        x[3] = np.random.randint(5,100)





clf = tree.DecisionTreeClassifier(criterion='entropy')#Classifier



clf.fit(X_train,y_train)



pred = clf.predict(X_test)

print(pred)

print("accuracy = ",clf.score(X_test,y_test))

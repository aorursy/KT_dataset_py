# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # graph plotting package



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
traindata = pd.read_csv('../input/train.csv')
traindata.head(10)
traindata.info()
traindata.describe()
def gender(sex):

    if sex == 'male':

        return 1

    elif sex == 'female':

        return 2

traindata['Gender'] = traindata['Sex'].apply(gender)
def address(embarked):

    if embarked == 'S':

        return 1

    elif embarked == 'C':

        return 2

    elif embarked == 'Q':

        return 3

traindata['Address'] = traindata['Embarked'].apply(address)
traindata = traindata.fillna(traindata.mean())

#traindata = traindata[np.isfinite(traindata['Age'])]
features = ['Pclass','Gender','SibSp','Parch','Fare','Age','Address']
X = traindata[features]

Y = traindata['Survived']

X.info()
X = X.values

Y = Y.values
from sklearn.cross_validation import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X, Y,test_size = 0.33,random_state=24)
plt.hist(ytrain)

plt.xlabel('Survivd rate', size=20)

plt.suptitle('Dsitribution of Survived Rate in the Training set')

plt.show()
from sklearn import tree
dt = tree.DecisionTreeClassifier(max_depth= 7, random_state= 24)
dt.fit(xtrain, ytrain)
print ('Train:', dt.score(xtrain, ytrain))

print ('Test:', dt.score(xtest, ytest))
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

y_pred = dt.predict(xtest)

survived = ['dead','survive']

matrix = pd.DataFrame(confusion_matrix(ytest,y_pred),columns=survived,index=survived)

print ('Confusion Matrix')

print (matrix, '\n')

print ('Classification Report')

print (classification_report(ytest,y_pred,target_names=survived),"\n")
#from IPython.display import Image

#import pydotplus 

#dot_data = tree.export_graphviz(dt, out_file=None)

#graph = pydotplus.graph_from_dot_data(dot_data)

#Image(graph.create_png())
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=300,max_depth=11).fit(xtrain,ytrain)
rf.predict(xtest)
print ('Train: ', rf.score(xtrain,ytrain))

print ('Test: ', rf.score(xtest,ytest))
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

y_pred = rf.predict(xtest)

survived = ['dead','survive']

matrix = pd.DataFrame(confusion_matrix(ytest,y_pred),columns=survived,index=survived)

print ('Confusion Matrix')

print (matrix, '\n')

print ('Classification Report')

print (classification_report(ytest,y_pred,target_names=survived),"\n")
testdata = pd.read_csv('../input/test.csv')
testdata.describe()
testdata['Gender'] = testdata['Sex'].apply(gender)

testdata['Address'] = testdata['Embarked'].apply(address)
testdata = testdata.fillna(traindata.mean())
testdata['Survived'] = rf.predict(testdata[features])
testdata.head()
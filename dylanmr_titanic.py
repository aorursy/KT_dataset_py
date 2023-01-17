# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from matplotlib import pyplot as plt
from sklearn import metrics
%matplotlib inline  

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
trainingdata = pd.read_csv("../input/train.csv")
testdata = pd.read_csv("../input/test.csv")
gendermodel = pd.read_csv("../input/gendermodel.csv")
genderclassmodel = pd.read_csv("../input/genderclassmodel.csv")
trainingcolumns = trainingdata.columns
testingcolumns = testdata.columns
gendercolumns = gendermodel.columns
genderclasscolumns = genderclassmodel.columns 
trainingcolumns
testdata
trainingdata = trainingdata.replace(['male','female'],[0,1])
trainingdata[['Age', 'Fare']].fillna(trainingdata.groupby('PassengerId')[['Age', 'Fare']].transform('mean'), inplace=True)
trainingdata[['Age', 'Fare']].fillna(trainingdata[['Age', 'Fare']].mean(), inplace=True)
testdata[['Age', 'Fare']].fillna(testdata.groupby('PassengerId')[['Age', 'Fare']].transform('mean'), inplace=True)
testdata[['Age', 'Fare']].fillna(testdata[['Age', 'Fare']].mean(), inplace=True)

pattern = '( \w[A-z].\w{,})'
titles = trainingdata.Name.str.extract(pattern)
titles.value_counts()
lower = titles.str.contains('Mr|Miss|Mrs|Ms')
upper = titles.str.contains('Master|Dr|Rev|Col|Major|Capt')
trainingdata['titles'] = trainingdata.Name
## lowerclass replacement
trainingdata.loc[lower,'titles'] = 0
trainingdata.loc[upper,'titles'] = 1
trainingdata

trainingdata = trainingdata[trainingdata['titles'].isin([0,1])]

#engineering test data
testtitles = testdata.Name.str.extract(pattern)
lower = testtitles.str.contains('Mr|Miss|Mrs|Ms')
upper = testtitles.str.contains('Master|Dr|Rev|Col|Major|Capt')
testdata['titles'] = testdata.Name
## lowerclass replacement
testdata.loc[lower,'titles'] = 0
testdata.loc[upper,'titles'] = 1
testdata = testdata[testdata['titles'].isin([0,1])]
testdata


X = trainingdata[['Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare', 'titles']]
Y = trainingdata['Survived']
X = X.fillna(X.Fare.mean())
test = testdata.replace(['male','female'],[0,1])
xtest = test[['PassengerId','Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare','titles']]
ytest = genderclassmodel[['PassengerId','Survived']]
xtest = xtest.fillna(xtest.mean().iloc[0])
xtest['Age'] = xtest['Age'].astype(int)
xtest = xtest.round({'Fare':2})
notintest = genderclassmodel[~genderclassmodel.isin(xtest)].dropna()
ytest = ytest[~ytest.isin(notintest)].dropna()
xtest = xtest.drop('PassengerId', axis = 1)
xtest
clf = RandomForestClassifier(n_estimators=1000, 
                             criterion='entropy')

clf90 = RandomForestClassifier(n_estimators=2100, 
                             criterion='gini', 
                             max_depth=4, 
                             min_samples_split=20, 
                             min_samples_leaf=40, 
                             min_weight_fraction_leaf=0.0,
                             max_features='auto', 
                             max_leaf_nodes=None, 
                             bootstrap=True, 
                             oob_score=False, 
                             n_jobs=-1, 
                             random_state=None, 
                             verbose=0, 
                             warm_start=False, 
                             class_weight='balanced')
clf90.fit(X, Y)
xtest = xtest[['Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare','titles']].astype(float)
ytest = ytest.astype(float)

clf90.score(xtest, ytest['Survived'])

from sklearn.metrics import confusion_matrix
predictarr = clf90.predict(xtest)
trutharr = ytest['Survived'].values
cm = confusion_matrix(trutharr, predictarr)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

confusion = np.matrix(cm_normalized)
print (cm_normalized)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
plt.show()




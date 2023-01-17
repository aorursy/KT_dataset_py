

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib 

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')

data.head()
data.info()
data.drop('Id',axis=1,inplace=True)
#cool visualization from https://www.kaggle.com/benhamner/python-data-visualizations



sns.pairplot(data, hue='Species', size=3)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



data['Species'] = LabelEncoder().fit_transform(data['Species'])

data.iloc[[0,1,-2,-1],:]
pipeline = Pipeline([

    ('normalizer', StandardScaler()), #Step1 - normalize data

    ('clf', LogisticRegression()) #step2 - classifier

])

pipeline.steps
#Seperate train and test data

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1].values,

                                                   data['Species'],

                                                   test_size = 0.4,

                                                   random_state = 10)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.model_selection import cross_validate



scores = cross_validate(pipeline, X_train, y_train)

scores
scores['test_score'].mean()
from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier



clfs = []

clfs.append(LogisticRegression())

clfs.append(SVC())

clfs.append(SVC())

clfs.append(KNeighborsClassifier(n_neighbors=3))

clfs.append(DecisionTreeClassifier())

clfs.append(RandomForestClassifier())

clfs.append(GradientBoostingClassifier())



for classifier in clfs:

    pipeline.set_params(clf = classifier)

    scores = cross_validate(pipeline, X_train, y_train)

    print('---------------------------------')

    print(str(classifier))

    print('-----------------------------------')

    for key, values in scores.items():

            print(key,' mean ', values.mean())

            print(key,' std ', values.std())
from sklearn.model_selection import GridSearchCV

pipeline.set_params(clf= SVC())

pipeline.steps
cv_grid = GridSearchCV(pipeline, param_grid = {

    'clf__kernel' : ['linear', 'rbf'],

    'clf__C' : np.linspace(0.1,1.2,12)

})



cv_grid.fit(X_train, y_train)
cv_grid.best_params_
cv_grid.best_estimator_
cv_grid.best_score_
y_predict = cv_grid.predict(X_test)

accuracy = accuracy_score(y_test,y_predict)

print('Accuracy of the best classifier after CV is %.3f%%' % (accuracy*100))
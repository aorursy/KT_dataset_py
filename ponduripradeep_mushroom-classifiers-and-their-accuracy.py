# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier

# Any results you write to the current directory are saved as output.
mushroom_df = pd.read_csv('../input/mushrooms.csv')
print(mushroom_df.shape)
pd.options.display.max_columns = None
mushroom_df.head()
# check whether the data has null values
mushroom_df.isnull().sum()
# As coloumns are categorical we need to change them to numerical
labelencoder = LabelEncoder()

for column in mushroom_df.columns:
    mushroom_df[column] = labelencoder.fit_transform(mushroom_df[column])
mushroom_df.head()
X = mushroom_df.drop('class',axis=1)     # as 'class' is the target variable
y = mushroom_df['class']

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

print('X_train:',x_train.shape)
print('X_test:',x_test.shape)

print('y_train: ',y_train.shape)
print('y_test:',y_test.shape)
classifiers = [LogisticRegression(),KNeighborsClassifier(n_neighbors=3),
               DecisionTreeClassifier(max_depth=5),SVC(kernel='linear',C=0.025),
              SVC(gamma=2,C=1),RandomForestClassifier(max_depth=5),
              BaggingClassifier(),GradientBoostingClassifier()]
classifier_names = ['Logistic Regression','KNearestNeighbours','DecisionTress',
                    'Linear SVM','RBF SVM','Random Forest',
                   'Bagging Classifier','Gradient Boosting Classifier']
classifier_compare_df = pd.DataFrame(columns=['Classifier','Mean Accuracy']) # empty dataframe for populating results
for classifier,name in zip(classifiers,classifier_names):
    model = classifier
    model.fit(x_train,y_train)
    mean_accuracy = model.score(x_test,y_test)
    classifier_compare_df.loc[len(classifier_compare_df)] = [name,mean_accuracy]
classifier_compare_df
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

print(train.shape)
train
print(train.shape)
print(train.columns)
print(train.info())

train.head()
train['Cover_Type'].value_counts()
train.describe()
array = [
    'Aspect', 
    'Slope', 
    'Horizontal_Distance_To_Hydrology', 
    'Vertical_Distance_To_Hydrology', 
    'Horizontal_Distance_To_Roadways',
    'Hillshade_9am',
    'Hillshade_Noon',
    'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points',
]

train[array].hist(bins=50, figsize=(20, 15))
plt.show()
print(np.sum(train[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4']]))
soil_type = [ 'Soil_Type' + str(i) for i in range(1, 41) ]
print(np.sum(train[soil_type]))
train.corr()['Cover_Type'].sort_values(ascending=False)
attributes = ['Cover_Type', 'Horizontal_Distance_To_Roadways', 'Slope', 'Hillshade_Noon']

scatter_matrix(train[attributes], figsize=(12, 8))
train_norm = train.copy()
test_norm = test.copy()
print(train_norm.shape, test_norm.shape)

for c in train.columns:
    s = train.dropna(subset=[c]).shape
    if s[0] < 15120:
        print(c)
        
# for c in test.columns:
#     s = test.dropna(subset=[c]).shape
#     if s[0] < 15120:
#         print(c)
array_scale = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']

scaler_list = []

for c in array_scale:
    scaler = MinMaxScaler()
    new_y = scaler.fit_transform(train_norm[c].values.reshape(-1, 1))
    new_y_test = scaler.transform(test_norm[c].values.reshape(-1, 1))
    train_norm[c] = new_y
    test_norm[c] = new_y_test
    
train_norm.head()
test_norm.head()
train_norm.corr()['Cover_Type'].sort_values(ascending=False)
lb = LabelBinarizer()
X = train_norm.drop(columns=['Id', 'Cover_Type'])
y = train_norm['Cover_Type']
lb.fit(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)
y_train_hot, y_test_hot = lb.transform(y_train.values.reshape(-1, 1)), lb.transform(y_test.values.reshape(-1, 1))

Xs = test_norm.drop(columns=['Id'])
def evaluate(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_pred, y_test))
tree_clf = DecisionTreeClassifier(random_state=18)
cols = ['Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Soil_Type29', 'Wilderness_Area1']

evaluate(tree_clf, X_train, y_train, X_test, y_test)
# test
ys = tree_clf.predict(Xs)
print(ys.shape)
print(ys)
test['Cover_Type'] = ys
test[['Id', 'Cover_Type']].head()
test[['Id', 'Cover_Type']].to_csv('forest_cover.csv', index=False)


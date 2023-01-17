# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import numpy package for arrays and stuff 

import numpy as np 



# import matplotlib.pyplot for plotting our result 

import matplotlib.pyplot as plt 



# import pandas for importing csv files 

import pandas as pd 
# import dataset

df = pd.read_csv('../input/tictactoe-endgame-dataset-uci/tic-tac-toe-endgame.csv')

df.head()
# Using Label Encoder convert catergorial data into Numerical data

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



df['V1'] = le.fit_transform(df['V1'])



df['V2'] = le.fit_transform(df['V2'])



df['V3'] = le.fit_transform(df['V3'])



df['V4'] = le.fit_transform(df['V4'])



df['V5'] = le.fit_transform(df['V5'])



df['V6'] = le.fit_transform(df['V6'])



df['V7'] = le.fit_transform(df['V7'])



df['V8'] = le.fit_transform(df['V8'])



df['V9'] = le.fit_transform(df['V9'])



df['V10'] = le.fit_transform(df['V10'])



df.head()
# Assigning values

X = df.iloc[:,0:9]

y = df.iloc[:,-1]
# import the Classifier

from sklearn.tree import DecisionTreeClassifier 



classifier=DecisionTreeClassifier(criterion='gini',max_depth = 6)



# fit the regressor with X and Y data 

classifier.fit(X,y)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=100)



y_pred_train=classifier.predict(X_train)



y_pred_test=classifier.predict(X_test)
# Confusion matrix

from sklearn.metrics import confusion_matrix

cm_train=confusion_matrix(y_train,y_pred_train)

cm_test=confusion_matrix(y_test,y_pred_test)



print(cm_train)

print(cm_test)
# Accuracy

from sklearn.metrics import accuracy_score

accuracy_train=accuracy_score(y_train,y_pred_train)

accuracy_test=accuracy_score(y_test,y_pred_test)



print(accuracy_train)

print(accuracy_test)
# Testing

negative_test=[0,1,1,0,2,1,2,1,1]

positive_test=[2,0,0,0,0,1,0,0,2]



test_np=[negative_test,positive_test]

classifier.predict(test_np)
# Graphviz

from sklearn.tree import export_graphviz



export_graphviz (classifier,out_file = 'tree_tic.dot',

                   feature_names = ['V1','V2','V3','V4','V5','V6','V7','V8','V9'])

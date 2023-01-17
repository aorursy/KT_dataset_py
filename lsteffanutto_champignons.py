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
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")
# print(df.head())
print("Shape: ", df.shape)
# print(df.columns)



from sklearn import preprocessing

labelencoder=preprocessing.LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])
    
print(df.head())


X = df[['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']]
Y = df["class"]

# X_dummy = pd.get_dummies(X) #convert letters in numerical values
# X_dummy.shape 
    
# print("all data features:",X.shape,"\n",X)
# print("column con etiquetas:",Y.shape,"\n",Y)
    
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

print("x_train:",x_train.shape,"\n",x_train)
print("x_test:",x_test.shape,"\n",x_test)
print("y_train:",y_train.shape,"\n",y_train)
print("y_test:",y_test.shape,"\n",y_test)
from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators= 100,max_features = 'sqrt',max_samples = 0.66 , bootstrap=True, random_state=42)
classifier.fit(x_train, y_train)

y_pred=classifier.predict(x_test)

# check precision, recall, F1 etc.

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("RESULTATS RANDOM FOREST CLASSIFIER\n ")

print("Accuracy RandomForest:",accuracy_score(y_test, y_pred) )
print("\nmatrix confusion:\n ",confusion_matrix(y_test,y_pred) )
print("\nrecall y F1:\n", classification_report(y_test,y_pred) )

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error

classifier_red= MLPClassifier(hidden_layer_sizes=(100,100,25),solver="sgd", max_iter=1000) # accuracy=0.999
# classifier_red= MLPClassifier(hidden_layer_sizes=(4,2),solver="sgd", max_iter=1000) # accuracy=0.5

classifier_red.fit(x_train, y_train)

y_pred_red=classifier_red.predict(x_test)

print("RESULTATS RED NEURONALES\n ")

print("Accuracy Red:",accuracy_score(y_test, y_pred_red) )
print("\nmatrix confusion:\n ",confusion_matrix(y_test,y_pred_red) )
print("\nrecall y F1:\n", classification_report(y_test,y_pred_red) )
print("Quadratic error Red:",mean_squared_error(y_test, y_pred_red) )
print('hello wolrd')
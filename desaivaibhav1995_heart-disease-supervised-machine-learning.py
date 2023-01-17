import numpy as np 

import pandas as pd 



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
heart = pd.read_csv("../input/heart.csv")
heart.head()
heart.shape
heart.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',

                 'rest_ecg', 'max_heart_rate_achieved','exercise_induced_angina', 'st_depression', 'slope',

                 'num_major_vessels', 'thalassemia', 'target']
heart.isnull().sum()
heart.dtypes
heart['target'].value_counts().plot(kind = 'bar', color = ['b','r'])
print("sex : {}".format(heart['sex'].nunique()))



print("chest_pain_type : {}".format(heart['chest_pain_type'].nunique()))



print("fasting_blood_sugar : {}".format(heart['fasting_blood_sugar'].nunique()))



print("rest_ecg : {}".format(heart['rest_ecg'].nunique()))



print("exercise_induced_angina : {}".format(heart['exercise_induced_angina'].nunique()))



print("slope : {}".format(heart['slope'].nunique()))



print("num_major_vessels : {}".format(heart['num_major_vessels'].nunique()))



print("thalassemia : {}".format(heart['thalassemia'].nunique()))
for col in ['sex','chest_pain_type','rest_ecg','exercise_induced_angina','fasting_blood_sugar','slope',

            'num_major_vessels','thalassemia']:

    heart[col] = heart[col].astype('category')
heart.dtypes
cols = ['sex','chest_pain_type','rest_ecg','exercise_induced_angina','fasting_blood_sugar','slope',

            'num_major_vessels','thalassemia', 'target']

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for c in cols:

    le.fit(list(heart[c].values))

    heart[c] = le.transform(list(heart[c].values)) 

    
heart = pd.get_dummies(heart, drop_first = True)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(heart.drop('target', 1), heart['target'],

                                                    test_size = .3, random_state=100) 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=5)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test,y_pred)

accuracy
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test,y_pred)

confusion_matrix
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(max_depth = 5, max_features = 4)

dec_tree.fit(X_train, y_train)
dec_tree_pred = dec_tree.predict(X_test)
dec_tree_accuracy = accuracy_score(y_test,dec_tree_pred)

dec_tree_accuracy
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights = 'distance', leaf_size = 20)

knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test,knn_pred)

knn_accuracy
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test,knn_pred)

confusion_matrix
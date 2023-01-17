import numpy as np

import pandas as pd



from sklearn.metrics import confusion_matrix



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split



from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score



import os

print(os.listdir("../input"))
dataset = pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
dataset.head()
columns = ['redshift', 'u', 'g', 'r', 'i', 'z', 'class']
dataset = dataset.loc[:, columns]



le = LabelEncoder().fit(dataset['class'])

dataset['class'] = le.transform(dataset['class'])
dataset.head()
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(labels = 'class', axis = 'columns'), dataset['class'], test_size = 0.3)
dici_param = {"activation": ["tanh", "logistic", "relu"]}

clf = GridSearchCV(estimator = MLPClassifier(max_iter=400), param_grid = dici_param, cv = 5, n_jobs = -1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
y_pred = clf.predict(X_test)
class_labels = le.inverse_transform([0,1,2])

confusion_df = pd.DataFrame(confusion_matrix(y_test, y_pred),

                            columns = class_labels,

                            index = class_labels)
confusion_df
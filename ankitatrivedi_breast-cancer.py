import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()

data.keys()
print(data['DESCR'])
df_data = pd.DataFrame(data['data'],columns=data['feature_names'])
df_data.head()
data['target_names']
from sklearn.model_selection import train_test_split
X = df_data

y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.3, random_state = 101)
from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))

print('\n')

print(classification_report(y_test,predictions))
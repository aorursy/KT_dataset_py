import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_digits
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

print('Libraries Imported')
import os
print(os.listdir('../input'))

data = pd.read_csv("../input/train.csv")
data.info()
print(len(data))
print(type(data))
data.describe()
data.head()
for i in range(0, 10):
    data[data['label']==i]['label'].hist(label=str(i))
plt.legend()

print("Visualization complete")
import random
from sklearn import ensemble
from sklearn.tree import DecisionTreeClassifier
type(data)
x = data[0:38000]
x = x.drop('label', axis=1)

#print(x)
y = data['label']
y = y[0:38000]
print("training data selected..")
#Random Forest
clf = ensemble.RandomForestClassifier()
clf.fit(x,y)
print("Model Trained..")
x_test = data[35000:]
x_test = x_test.drop('label', axis=1)
y_test = data['label']
y_test = y_test[35000:]
print(len(y_test))
print("Test Data Selected")

clf.predict(x_test)
print("Prediction completed")
score = clf.score(x_test, y_test)
rs = "Score:" + str(score*100) + " %"
print(rs)
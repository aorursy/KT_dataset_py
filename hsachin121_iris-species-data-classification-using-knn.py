##import all the libraries and framework

import numpy as np

import pandas as pd
## loading a dataset



iris_data = pd.read_csv('../input/Iris.csv')
iris_data.head(5)      ##explore head of data
iris_data.shape
iris_data.describe()
iris_data.groupby('Species').size()
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']
X = iris_data[features].values
y= iris_data['Species'].values
y
from sklearn.preprocessing import LabelEncoder ##import sklearn LabelEncoder
la_en = LabelEncoder()
y = la_en.fit_transform(y)  ### encoding string values to numbers
print(set(y.tolist()))  ### now we can see that we have only three labels or classes
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure()

sns.pairplot(iris_data.drop("Id", axis=1), hue = "Species", size=5, markers=["o", "s", "D"])

plt.show()
### import sklearn libraries
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
y_pred
accuracy = accuracy_score(y_test,y_pred)*100

print('accuracy = ' , accuracy ,'%' )
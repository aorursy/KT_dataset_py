# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
iris_filepath = '/kaggle/input/iris/Iris.csv'

iris_data = pd.read_csv(iris_filepath)
iris_data.describe()

iris_data.head()
iris_data.Species
iris_data.dtypes
# 1 -> Iris-setosa

# 2 -> Iris-virginica

# 3 -> Iris-versicolor

iris_data['Species'] = iris_data['Species'].str.replace('Iris-setosa','1')

iris_data['Species'] = iris_data['Species'].str.replace('Iris-virginica','2')

iris_data['Species'] = iris_data['Species'].str.replace('Iris-versicolor','3')
iris_data['Species'] = iris_data['Species'].astype(float)
iris_data.columns
#data visualization

plt.figure(figsize=(10,10))

sns.barplot(x = iris_data['Species'], y = iris_data['PetalLengthCm'])
sns.regplot(x = iris_data['Species'], y = iris_data['PetalLengthCm'])
sns.regplot(x = iris_data['Species'], y = iris_data['SepalWidthCm'])
#prediction model 

y = iris_data.Species

X = iris_data
from sklearn.tree import DecisionTreeClassifier

iris_model = DecisionTreeClassifier(random_state = 1)

iris_model.fit(X,y)
print("making predictions on first five flowers: ")

print(X)

print("\n the predictions are :")

print(iris_model.predict(X))
#model validation

#since this was only the test data there seems to be no error

from sklearn.metrics import mean_absolute_error

predicted_iris = iris_model.predict(X)

mean_absolute_error(y, predicted_iris)
#introducing train test split

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)



new_model = DecisionTreeClassifier()



new_model.fit(train_X, train_y)
val_predictions = new_model.predict(val_X)

print(mean_absolute_error(val_y, val_predictions))
from sklearn import metrics



print('The accuracy of the Decision Tree is',metrics.accuracy_score(val_predictions,val_y))
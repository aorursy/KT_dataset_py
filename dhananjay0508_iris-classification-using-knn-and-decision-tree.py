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
iris_data = pd.read_csv('../input/iris/Iris.csv')
iris_data.head()
iris_data.shape
import seaborn as sns
%matplotlib inline
iris_data['Species'].value_counts()
iris_data.isnull().sum()
ax = sns.boxplot(x = 'Species', y = 'PetalWidthCm',data = iris_data)
ax = sns.boxplot(x = 'Species', y = 'PetalLengthCm',data = iris_data)
iris_data.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

iris_data.columns
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = iris_data[features]
X.head()
y = iris_data['Species']
y
X_trian,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.33,shuffle = True)
classification = DecisionTreeClassifier()
classification.fit(X_trian,y_train)
prediction = classification.predict(X_test)
prediction[:10]
y_test[:10]
accuracy_score(y_true = y_test, y_pred = prediction)*100
classification1 = KNeighborsClassifier()
classification1.fit(X_trian,y_train)
prediction1 = classification1.predict(X_test)
prediction1
y_test
accuracy_score(y_true = y_test, y_pred = prediction1)*100

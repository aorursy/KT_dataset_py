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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
dataset = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv') # reading the dataset
dataset.head()
from sklearn.model_selection import train_test_split
from sklearn import metrics
X = dataset.iloc[:, 2:-1]
X_column_name = X.columns
X.head()
Y = dataset.iloc[:, 1:2]
Y.head()
Y['diagnosis'].unique()
who = {'M': 0, 'B': 1}
Y['diagnosis'] = Y['diagnosis'].map(who)
Y.head()
from sklearn.preprocessing import StandardScaler
scalar_x = StandardScaler()
X = scalar_x.fit_transform(X)
X = pd.DataFrame(data = X, columns = X_column_name)
X.head()
from sklearn.decomposition import PCA
pca = PCA(n_components = 2, random_state = 42)
pca.fit(X)
x_pca = pca.transform(X)
x_pca.shape
Y.shape
x_pca = pd.DataFrame(data = x_pca)
x_pca.head()
from sklearn.model_selection import train_test_split            # Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x_pca, Y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
def logistic(x_train, y_train, x_test, y_test):
    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(f'Accuracy of test set: {accuracy_score(y_test, y_pred)}')
%%time
logistic(x_train, y_train, x_test, y_test)        # With PCA
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(x_train, y_train)
y_predf = classifier.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predf)
from sklearn.model_selection import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size = 0.25, random_state = 0)
%%time
logistic(x_train1, y_train1, x_test1, y_test1)        # Without PCA


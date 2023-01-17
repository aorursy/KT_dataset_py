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
from sklearn.tree import DecisionTreeClassifier

ir_data = '../input/mydataset/IRIS.csv'
iris_data = pd.read_csv(ir_data)
y = iris_data.species
feature_columns = ['sepal_length','sepal_width','petal_length','petal_width']

X = iris_data[feature_columns]




iris_data = DecisionTreeClassifier()

iris_data.fit(X, y)

pred = iris_data.predict(X)
pred
from sklearen.metrics import accuracy_score,classification_report
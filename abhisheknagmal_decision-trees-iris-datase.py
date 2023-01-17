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

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.preprocessing import LabelEncoder
df =pd.read_csv('../input/iris-dataset/Iris.csv',encoding="ISO-8859-1")
df.head()
df.isnull().sum().any()
df.info()
df.describe()
df.Species.value_counts()
df.Species=LabelEncoder().fit_transform(df.Species)
X = df.iloc[:,:-1]

y =df['Species']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=12)
model = DecisionTreeClassifier(random_state=12)

model.fit(X_train,y_train)
y_pred= model.predict(X_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
!pip install pydotplus
from sklearn import tree

from IPython.display import Image

import pydotplus
dot_data= tree.export_graphviz(model,filled=True,feature_names=X_train.columns)

graph= pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
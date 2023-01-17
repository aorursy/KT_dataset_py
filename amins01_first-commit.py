import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from IPython.display import Image



# Pandasの設定をします

pd.set_option('chained_assignment', None)



# matplotlibのスタイルを指定します。これでグラフが少しかっこよくなります。

plt.style.use('ggplot')

plt.rc('xtick.major', size=0)

plt.rc('ytick.major', size=0)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.tail()
x = df_train['Sex']

y = df_train['Survived']
x_test = df_test['Sex']

y_test_pred = x_test.map({'female': 1, 'male': 0}).astype(int)
df_kaggle = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived':np.array(y_test_pred)})

df_kaggle.to_csv('kaggle_gendermodel.csv', index=False)
df_kaggle.head()
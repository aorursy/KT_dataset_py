import sys

print(sys.version)
from subprocess import check_output

print(check_output(["ls", "../input/titanic"]).decode("utf8"))
from IPython.display import Image

Image(url='http://graphics8.nytimes.com/images/section/learning/general/onthisday/big/0415_big.gif')
%matplotlib inline
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
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')



df_train.tail()
df_test.tail()
df_train.groupby('Survived').count()
df_test.count()
x = df_train['Sex']

y = df_train['Survived']

z = df_train['PassengerId']

x.head()
y.head()
z.head()
y_pred = x.map({'female': 1, 'male': 0}).astype(int)

y_pred.head()
print('Accuracy: {:.3f}'.format(accuracy_score(y, y_pred)))
print(classification_report(y, y_pred))
cm = confusion_matrix(y, y_pred)

print(cm)
#genderモデルでテストデータに対して予測してsubbmit

x2 = df_test['Sex']

#y2 = df_train['Survived'] #テストデータには生存フラグはない。ここを予測する

z2 = df_test['PassengerId']

x2.head()
z2.head()
y2_pred = x2.map({'female': 1, 'male': 0}).astype(int)

y2_pred.head()
#採点用のデータ作成

df_kaggle = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived':np.array(y2_pred)})

df_kaggle
#outputファイル作成

df_kaggle.to_csv('kaggle_my_gender.csv', index=False)
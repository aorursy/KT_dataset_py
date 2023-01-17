# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
tsd = pd.read_csv('../input/thoraric-surgery/ThoraricSurgery.csv', index_col = 'id')
tsd.head()
tsd.info()
tsd.dtypes
tsd[['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']] = \
(tsd[['PRE7', 'PRE8', 'PRE9', 'PRE10', 'PRE11', 'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32', 'Risk1Yr']] == 'T').astype(int)
tsd['DGN']=tsd['DGN'].str[-1:].astype(int)
tsd['PRE6']=tsd['PRE6'].str[-1:].astype(int)
tsd['PRE14']=tsd['PRE14'].str[-1:].astype(int)

tsd.info()
tsd.head()
tsd.describe()
new_headers = ['диагнос','емкость_легких','объем_выдоха','шкала_зуброда','боль','кровохарканье','одышка',
       'кашель','слабость','размер_опухоли','диабет','ИМ','заболевания_периферических_артерий','курение','астматик','возраст','риск_год']
tsd.columns = new_headers
tsd.hist()
plt.subplots(figsize=(10,10))
sns.heatmap(tsd.corr(), square=True)
plt.show()
fig, ax = plt.subplots(figsize=(12, 12))
mask=np.zeros_like(tsd.corr())
mask[np.triu_indices_from(mask)] = True
sns.heatmap(tsd.corr(), annot=True, linewidths=.1, cmap="YlGnBu", square=True, mask=mask, cbar=False)
sns.pairplot(tsd[['диагнос','размер_опухоли','объем_выдоха','курение', 'возраст', 'астматик', 'риск_год']], 
             hue='риск_год', diag_kws={'bw':1.5}, height=3)
fig, ax = plt.subplots(figsize = (10,6))
sns.barplot(x= 'диагнос', y='риск_год', 
            data = tsd, palette="Blues_d",
            ax=ax, ci=None)
tsd[tsd.диагнос.isin([1, 6])]
fig, ax = plt.subplots(figsize = (10,6))
sns.distplot(tsd.диагнос, kde=False)
tsd1 = tsd.copy()
tsd1['диагнос'] = np.where(tsd['диагнос'].isin([1,5,6,7,8]), 0, tsd['диагнос'])
tsd1.head()
fig, ax = plt.subplots(figsize = (10,6))
sns.distplot(tsd1.диагнос, kde=False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import metrics
from imblearn.over_sampling import SMOTE, ADASYN
import warnings
warnings.filterwarnings("ignore")
X = tsd1.drop(columns='диагнос')
y = tsd1.диагнос
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=99, stratify=y
)
print(metrics.classification_report(y_test, predictions))
pd.Series(y_train).value_counts()

pd.Series(y_train_2).value_counts()
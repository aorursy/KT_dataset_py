import numpy as np  # linear algebra

import pandas as pd  # read and wrangle dataframes

import matplotlib.pyplot as plt # visualization

import seaborn as sns # statistical visualizations and aesthetics

from sklearn.base import TransformerMixin # To create new classes for transformations

from sklearn.preprocessing import (FunctionTransformer, StandardScaler) # preprocessing 

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from sklearn.decomposition import PCA # dimensionality reduction

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from scipy.stats import boxcox # data transform

from sklearn.model_selection import (train_test_split, KFold , StratifiedKFold, 

                                     cross_val_score, GridSearchCV, 

                                     learning_curve, validation_curve) # model selection modules

from sklearn.pipeline import Pipeline # streaming pipelines

from sklearn.base import BaseEstimator, TransformerMixin # To create a box-cox transformation class

from collections import Counter

import warnings

# load models

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from xgboost import (XGBClassifier, plot_importance)

from sklearn.svm import SVC

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from time import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

import seaborn as sns

from sklearn.compose import ColumnTransformer

from sklearn.metrics import confusion_matrix



%matplotlib inline 

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/crystal-system-properties-for-liion-batteries/lithium-ion batteries.csv")

df.head()
df.describe()
df['Materials Id']=df['Materials Id'].str.extract(r"(\d{6})")

df['Materials Id']
plt.figure(figsize=(12, 10))

corr_matrix = df.corr()

sns.heatmap(corr_matrix, lw=0.5, cmap='coolwarm', annot=True)
#df['Has Bandstructure']=df['Has Bandstructure'].astype(int)

df = pd.concat([df,pd.get_dummies(df['Has Bandstructure'],prefix="Has Bandstructure")],axis=1)

df.drop('Has Bandstructure', axis=1, inplace=True)


#df['Crystal System']=df['Crystal System'].replace(to_replace="monoclinic",value="1")

#df['Crystal System']=df['Crystal System'].replace(to_replace="orthorhombic",value="2")

#df['Crystal System']=df['Crystal System'].replace(to_replace="triclinic",value="3")



#crystal_df = pd.get_dummies(data['Crystal System'],columns=["monoclinic","orthorhombic","triclinic"] )

#crystal_df
df = pd.concat([df,pd.get_dummies(df['Spacegroup'],prefix="Spacegroup")],axis=1)

df.drop('Spacegroup', axis=1, inplace=True)
df = pd.concat([df,pd.get_dummies(df['Formula'],prefix="Formula")],axis=1)

df.drop('Formula', axis=1, inplace=True)
# Replace using median 

xmedian = df.median()

df.fillna(xmedian, inplace=True)
X = df.drop('Crystal System', axis=1)

y = df['Crystal System']
#df=data.drop('Formula', axis=1)

X
#data
X.columns

X.shape
y.shape
#data = data.drop(['Materials Id', 'Volume', 'Nsites', 'Density (gm/cc)'], axis=1)

#np.where(X.values >= np.finfo(np.float64).max)

#df.fillna(df.mean(), inplace=True)
X.isnull().values.any()
# Replace using median 

#xmedian = X.median()

#X.fillna(xmedian, inplace=True)
y.isnull().values.any()
from sklearn.decomposition import PCA

pca = PCA()

pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=0.2, shuffle=True)
X
y
print(X_train)

print(y_train)
import xgboost as xgb

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from graphviz import *
xg_cl = xgb.XGBClassifier(objective='multi:softmax', n_estimators=200,seed=123,learning_rate=0.15,max_depth=5,colsample_bytree=1,subsample=1)
xg_cl.fit(X_train,y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))
#XGBoost in the package itself gives us the feature importance to understand how each features compares

xgb.plot_importance(xg_cl)

plt.show()
confusion_matrix(y_test, preds, normalize='all')
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, preds)

plt.figure(figsize = (14,10))

sns.heatmap(cm, annot=True)
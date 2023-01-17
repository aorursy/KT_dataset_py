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
mice=pd.read_csv('/kaggle/input/mice-protein-expression/Data_Cortex_Nuclear.csv')

mice
stats=mice.describe()

stats
mice['Genotype'].value_counts()
mice['Treatment'].value_counts()
mice['Behavior'].value_counts()
mice['class'].value_counts()
mice.corr()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(30,20))

sns.heatmap(mice.corr(),annot=True,linewidth=0.5)
mice.isnull().sum()
mice
pd.set_option('display.max_columns', None)

mice.head(1)
mice.dropna()
pd.set_option('display.max_rows', None)

mice.dtypes
mice[['MouseID','Genotype','Treatment','Behavior','class']].isnull().sum()
numeric=mice.drop(columns=['MouseID','Genotype','Treatment','Behavior','class'])

categorical=mice[['MouseID','Genotype','Treatment','Behavior','class']]
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

mice_filled = my_imputer.fit_transform(numeric)

df=pd.DataFrame(mice_filled)

df.columns=numeric.columns
combined=categorical.join(df)

combined.describe()
mice.describe()
import sklearn as sk



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
X=combined.loc[:,'DYRK1A_N':'CaNA_N']

y=combined['class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
clf = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=1)

clf.fit(X_train, y_train)
clf.feature_importances_
y_pred=clf.predict(X_test)

accuracy_score(y_test, y_pred)
lr=LogisticRegression()



# fitting the model to the training data

lr.fit(X_train, y_train)



# use the model to predict on the testing data

lr.predict(X_test)



# Printing the accuracy of the model

score = lr.score(X_test, y_test)

score
from sklearn.neighbors import KNeighborsClassifier



#Making a new model to predict genotypes to make an ROC curve

kgmodel = KNeighborsClassifier(n_jobs=-1)

kgmodel.fit(X_train, y_train)



#Getting the accuracy

pred_gen = kgmodel.predict(X_test)

accuracy_score(pred_gen, y_test)
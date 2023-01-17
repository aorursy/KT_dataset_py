# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn 
import matplotlib.pyplot as graphic
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# test and train database
df = pd.read_csv('../input/adult-pmr3508/train_data.csv')
testAdult = pd.read_csv('/kaggle/input/adult-pmr3508/train_data.csv')

df.head(10)
df.info()
df.describe()
df_set = df.copy()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df_set['income'] = le.fit_transform(df_set['income'])
#We can see that the income has only binary data.
df_set.head()
mask = np.triu(np.ones_like(df_set.corr(), dtype=np.bool))

graphic.figure(figsize=(10,10))

sns.heatmap(df_set.corr(), mask=mask, square = True, annot=True, vmin=-1, vmax=1, cmap='autumn')
graphic.show()
sns.catplot(x="income", y="hours.per.week",kind="boxen", data=df_set);
# It is possible to conclude that more time working imply a high income
sns.catplot(x="income", y="education.num", kind="boxen", data=df_set);
# It is possible to conclude that a better study level imply a high income
sns.catplot(x="income", y="age", kind="boxen", data=df_set);
#The age is a important fact to the income, as we can see below
sns.catplot(y="sex", x="income", kind="bar", data=df_set);
# The graph below show a sad fact about our society, while 30% of the Male earn more than 50k annualy.
# Only 10% of the female earn that.
sns.catplot(y="workclass", x="income", kind="bar", data=df_set);
sns.catplot(y="marital.status", x="income", kind="bar", data=df_set);
# The Marrieds has a higher indice of income
sns.catplot(y="occupation", x="income", kind="bar", data=df_set);
sns.catplot(y="native.country", x="income", kind="bar", data=df_set);
# Since more than 90% of the native country of dataset information is from United-States the others country has a high std in the data
sns.catplot(y="race", x="income", kind="bar", data=df_set);
# The race is another important fact in the income, White and Asian earns the highest income.

Y_train = df.pop('income')

X_train = df

X_train.head()
X_train = preprocessor.fit_transform(X_train)
knn=KNeighborsClassifier()
score_knn=cross_val_score(knn, X_train,Y_train, cv=5, n_jobs=-1)
print("Best accuracy:",max(score_knn))
score_knn.mean()
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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_excel('/kaggle/input/covid19/dataset.xlsx')

df.head()
original_columns = df.columns.values



columns =['ID']



for i in range(110):

    columns.append('V{}'.format(i))



df.columns = columns



df = df[columns[1:]]



df.head()
df['V1'] = [0 if a == 'negative' else 1 for a in df['V1'].values]
sns.violinplot(x="V0", y='V1', data=df, vert=False)
sns.distplot(df['V1'])
corrmat = df.corr()

f, ax = plt.subplots(figsize=(24, 18))

sns.heatmap(corrmat, vmax=.8, square=True);
corrmat_v1 = corrmat.nlargest(10, 'V1')



features = corrmat_v1.index.values.tolist()



sns.heatmap(df[features].corr(), yticklabels=features, xticklabels=features, square=True);



#sns.heatmap(corrmat_best)
from sklearn import tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer



X = df[features[1:]]

Y = df[features[0]]



imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)

imp = imp.fit(X)



X = pd.DataFrame(imp.transform(X), columns=features[1:])



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)



model_tree = tree.DecisionTreeClassifier()

model_tree = model_tree.fit(X_train, y_train)



y_pred = model_tree.predict(X_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
fig, ax = plt.subplots(figsize=(15,15))

tm = tree.plot_tree(model_tree, ax=ax)

plt.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer



X = df[features[1:]]

Y = df[features[0]]



imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)

imp = imp.fit(X)



X = pd.DataFrame(imp.transform(X), columns=features[1:])



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)



model_rf = RandomForestClassifier(max_depth=2, random_state=0)

model_rf = model_rf.fit(X_train, y_train)



y_pred = model_rf.predict(X_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer



X = df[features[1:]]

Y = df[features[0]]





imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)

imp = imp.fit(X)



X = pd.DataFrame(imp.transform(X), columns=features[1:])



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3)



model_xgb = XGBClassifier()



model_xgb.fit(X_train, y_train)



y_pred = model_xgb.predict(X_test)



predictions = [round(value) for value in y_pred]



accuracy = accuracy_score(y_test, predictions)



print("Accuracy: %.2f%%" % (accuracy * 100.0))
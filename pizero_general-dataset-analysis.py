# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



sns.set_context('poster')

sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/xAPI-Edu-Data.csv')

df.head()
# Let's see if there are missing values

df.isnull().mean()
sns.countplot(x='gender', data=df, order=['M','F'],palette='bright')

plt.show()
ax = sns.countplot(x='Class', data=df, order=['L', 'M', 'H'],palette='bright')

for p in ax.patches:

    ax.annotate('{:.2f}%'.format((p.get_height() * 100) / len(df)), (p.get_x() + 0.24, p.get_height() + 2))
sns.countplot(x='gender', hue='Relation', data=df, order=['M','F'],palette='bright')

plt.show()
sns.countplot(x='Class', hue='gender', data=df, order=['L', 'M', 'H'],palette='bright')

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder
df_train = df.copy()



label = LabelEncoder()

cat_Colums = df_train.dtypes.pipe(lambda df_train: df_train[df_train=='object']).index

for col in cat_Colums:

    df_train[col] = label.fit_transform(df_train[col])



X = df_train[df_train.columns.difference(['Class'])]

y = df_train['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
classifier = DecisionTreeClassifier(random_state=1990, criterion='gini', max_depth=3)

classifier.fit(X_train, y_train)
X_test.axes
"""index_record = 477

print('Expected:', y_test.ix[index_record])

print('Predicted: ', classifier.predict([X_test.ix[index_record]]))"""
from sklearn.model_selection import cross_val_score

scores_dt = cross_val_score(classifier, X_test, y_test, scoring='accuracy', cv=5)

print('Acuracy: {0:.2f}%'.format(scores_dt.mean() * 100))
classifier.fit(df_train[df_train.columns.difference(['Class'])], df_train['Class'])



features_importance = zip(classifier.feature_importances_, X_train.columns)

for importance, feature in sorted(features_importance, reverse=True):

    print("%s: %.2f%%" % (feature, importance * 100))
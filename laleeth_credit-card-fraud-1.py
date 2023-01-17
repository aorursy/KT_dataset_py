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
df=pd.read_csv('../input/creditcardfraud/creditcard.csv')
df.head(2)
df.shape
import matplotlib.pyplot as plt

import seaborn as sns
df.describe()
df.isna().sum()
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
colors = ['#22EA52','#EA225F']

sns.countplot('Class',data=df,palette=colors)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold

X = df.drop('Class', axis=1)

y = df['Class']



skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in skf.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
plt.figure(figsize=(40,20))

sns.heatmap(df.corr(),annot=True,annot_kws={"size": 10})

plt.show()
sns.boxplot(x="Class", y="V17", data=df, palette=colors)

sns.boxplot(x="Class", y="V14", data=df, palette=colors)
sns.boxplot(x="Class", y="V12", data=df, palette=colors)
sns.boxplot(x="Class", y="V10", data=df, palette=colors)
sns.boxplot(x="Class", y="V18", data=df, palette=colors)
from sklearn.model_selection import train_test_split



# This is explicitly used for undersampling.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.values

X_test = X_test.values

y_train = y_train.values

y_test = y_test.values
X_train.shape,X_test.shape,y_train.shape,y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



classifiers = {

    "LogisiticRegression": LogisticRegression(),

    "KNearest": KNeighborsClassifier(),

}

from sklearn.model_selection import cross_val_score





for key, classifier in classifiers.items():

    classifier.fit(X_train, y_train)

    training_score = cross_val_score(classifier, X_train, y_train, cv=5)

    print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")

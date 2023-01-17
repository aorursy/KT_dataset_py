import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
white_wine = pd.read_csv('../input/winequality-white.csv')
white_wine.head()
white_wine.info()
white_wine.apply(lambda x: sum(x.isnull()), axis=0)
white_wine['quality'].unique()
white_wine['quality'].value_counts()
sns.countplot(x='quality', data=white_wine)
def classify_wine_quality(quality):

    if quality < 6:

        return 'poor quality'

    elif quality == 6:

        return 'normal quality'

    else:

        return 'excellent quality'

    

white_wine['quality classification'] = white_wine['quality'].apply(classify_wine_quality)
sns.set_style('darkgrid')

sns.set(rc={'figure.figsize':(10,5)})

sns.countplot(x='quality classification', data=white_wine)
sns.barplot(x='quality', y='fixed acidity', data=white_wine)
sns.barplot(x='quality', y='pH', data=white_wine)
sns.barplot(x='quality', y='sulphates', data=white_wine)
sns.barplot(x='quality', y='alcohol', data=white_wine)
sns.set(rc={'figure.figsize':(10,5)})

sns.boxplot(y='alcohol', x='quality classification', hue='quality classification', data=white_wine)
sns.boxplot(y='pH', x='quality classification', data=white_wine, hue='quality classification')
sns.boxplot(y='sulphates', x='quality classification', data=white_wine, hue='quality classification')
sns.boxplot(y='fixed acidity', x='quality classification', data=white_wine, hue='quality classification')
correlation = white_wine.corr()['quality'].drop('quality')

print(correlation.sort_values(kind='quicksort', ascending=False))
#use LabelEncoder for the categorigal values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

white_wine['quality classification'] = le.fit_transform(white_wine['quality classification'])

white_wine['quality classification'].dtypes
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(white_wine.drop(['quality classification', 'quality'], axis=1 ), white_wine['quality classification'], test_size=0.40)
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
print(classification_report(predictions, y_test))
print(confusion_matrix(predictions, y_test))
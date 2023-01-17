# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Import and read the CSV

heart = pd.read_csv('../input/heart.csv')

heart.head()
# Look at basic information for the dataset

heart.info()
heart.describe()
# Check for Null values. (None found)

sns.heatmap(heart.isnull(), yticklabels = False, cmap = 'viridis')
# Countplot | target | 1 = Presence of heart disease

sns.set_style('whitegrid')

sns.countplot(x = 'target', data = heart, palette =  'RdBu_r')
# Countplot | sex | 0 = female, 1 = male

sns.set_style('whitegrid')

sns.countplot(x = 'sex', order = [1, 0], data = heart, palette ='RdBu_r')
# Countplot | target | hue = sex | 1 = Presence of heart disease

sns.countplot( x = 'target', data = heart, hue = 'sex', hue_order = [1, 0], palette = 'RdBu_r')

l = plt.legend()

l.get_texts()[0].set_text('Male')

l.get_texts()[1].set_text('Female')
# Countplot | target | hue = cp (chest pain) | 1 = Presence of heart disease

sns.countplot( x = 'target', data = heart, hue = 'cp', palette = 'magma')

l = plt.legend()

l.get_texts()[0].set_text('Typical angina')

l.get_texts()[1].set_text('Atypical angina')

l.get_texts()[2].set_text('Non anginal pain')

l.get_texts()[3].set_text('Asymtomatic')
# Boxplot | target vs. trestbps (Resting Heart Beat) | hue = sex | 1 = Presence of heart disease

sns.boxplot(x = 'target', y = 'trestbps', hue = 'sex', hue_order = [1, 0], palette = 'RdBu_r', data = heart)

l = plt.legend()

l.get_texts()[0].set_text('Male')

l.get_texts()[1].set_text('Female')
# Boxplot | target vs. chol (cholestoral) | hue = sex | 1 = Presence of heart disease

sns.boxplot(x = 'target', y = 'chol', hue = 'sex', hue_order = [1, 0], palette = 'RdBu_r', data = heart)

l = plt.legend()

l.get_texts()[0].set_text('Male')

l.get_texts()[1].set_text('Female')
# Countplot | target | hue = exang (Exercise Induced Angina) | 1 = Presence of heart disease

sns.countplot( x = 'target', data = heart, hue = 'exang', palette = 'magma')

l = plt.legend()

l.set_title('Angina')

l.get_texts()[0].set_text('Non-Exercise Induced')

l.get_texts()[1].set_text('Exercise Induced')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(heart.drop('target',axis=1), 

                                                    heart['target'], test_size=0.30)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver = 'lbfgs', max_iter = 1000)



logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)



from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
X_train, X_test, y_train, y_test = train_test_split(heart.drop(['target', 'trestbps'], axis=1), 

                                                    heart['target'], test_size=0.30)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver = 'lbfgs', max_iter = 1000)



logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)



from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
X_train, X_test, y_train, y_test = train_test_split(heart.drop(['target', 'trestbps', 'chol'], axis=1), 

                                                    heart['target'], test_size=0.30)
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression(solver = 'lbfgs', max_iter = 1000)



logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)



from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))



from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
4
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
import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as st

%matplotlib inline



sns.set(style="whitegrid")
import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


print('The shape of the dataset : ', df.shape)
df.head(5)
df['thal'].value_counts()
df.describe()



df['target'].value_counts()
sns.countplot(x="target", data=df)

plt.show()
df.groupby('sex')['target'].value_counts()
sns.catplot(x="target", col="sex", data=df, kind="count", height=5, aspect=1)

plt.show()
df.groupby('fbs')['target'].value_counts()
sns.countplot(x="target", hue="fbs", data=df)

plt.show()
df.groupby('exang')['target'].value_counts()
sns.countplot(x="target", hue="exang", data=df)

plt.show()
correlation = df.corr()

correlation['target'].sort_values(ascending=False)
df['cp'].value_counts()
sns.countplot(x="cp", data=df)

plt.show()
df.groupby('cp')['target'].value_counts()
sns.catplot(x="target", col="cp", data=df, kind="count", height=8, aspect=1)

plt.show()


x = df['thalach']

sns.distplot(x, bins=10)

plt.show()


sns.boxplot(x="target", y="thalach", data=df)

plt.show()
df['oldpeak'].nunique()
sns.boxplot(x="target", y="oldpeak", data=df)

plt.show()
y = df.target.values

x_data = df.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
"""Building machine learning models: 

We will try 10 different classifiers to find the best classifier after tunning model's hyperparameters that will best generalize the unseen(test) data."""

seed =6

'''Now initialize all the classifiers object.'''

'''#1.Logistic Regression'''

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()



'''#2.Support Vector Machines'''

from sklearn.svm import SVC

svc = SVC(gamma = 'auto')



'''#3.Random Forest Classifier'''

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state = seed)



'''#4.KNN'''

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



'''#5.Gaussian Naive Bayes'''

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()



'''#6.Decision Tree Classifier'''

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state = seed)



'''#7.Gradient Boosting Classifier'''

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(random_state = seed)

'''#8.Adaboost Classifier'''

from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(random_state = seed)



'''#9.ExtraTrees Classifier'''

from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier(random_state = seed)



'''#10.Extreme Gradient Boosting'''

from xgboost import XGBClassifier

xgbc = XGBClassifier(random_state = seed)
from sklearn.model_selection import cross_val_score

def train_accuracy(model):

    #model.fit(x_train, y_train)

    scores = cross_val_score(model, x, y, cv=5)

    train_accuracy = scores.mean()

    train_accuracy = np.round(train_accuracy*100, 2)

    return train_accuracy





'''Models with best training accuracy:'''

train_accuracy = pd.DataFrame({'Train_accuracy(%)':[train_accuracy(lr), train_accuracy(svc), train_accuracy(rf), train_accuracy(knn), train_accuracy(gnb), train_accuracy(dt), train_accuracy(gbc), train_accuracy(abc), train_accuracy(etc), train_accuracy(xgbc)]})

train_accuracy.index = ['LR', 'SVC', 'RF', 'KNN', 'GNB', 'DT', 'GBC', 'ABC', 'ETC', 'XGBC']

sorted_train_accuracy = train_accuracy.sort_values(by = 'Train_accuracy(%)', ascending = False)

sorted_train_accuracy
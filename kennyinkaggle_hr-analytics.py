# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_df = pd.read_csv("../input/hr-data-for-analytics/HR_comma_sep.csv")

data_df.info()
data_df.describe()
def distplay(feature):

    sns.distplot(data_df[feature])

    

def catplay(feature):

    cat_counts = data_df[feature].value_counts()

    sns.barplot(cat_counts.index, cat_counts)

    print("Mean of %s: %.3f" % (feature, data_df[feature].mean()))

    print("Mode of %s: %.3f" % (feature, data_df[feature].mode()))
distplay('satisfaction_level')



# --> most of staff is satisfied with the job.
distplay('last_evaluation')
catplay('number_project')
distplay('average_montly_hours')
catplay('time_spend_company')
catplay('Work_accident')
catplay('left')
catplay('promotion_last_5years')
cat_counts = data_df['salary'].value_counts()

sns.barplot(cat_counts.index, cat_counts)
cat_counts = data_df['sales'].value_counts()

plt.figure(figsize=(12,8))

sns.barplot(cat_counts.index, cat_counts)
cor = data_df.corr()

sns.heatmap(cor, annot=True)
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
data_copy = data_df.copy()
col_list = ['satisfaction_level', 'last_evaluation']



for col in col_list:

    data_copy[col] = StandardScaler().fit_transform(data_copy[col].values.reshape(-1,1)).reshape(1,-1)[0]
col = ['average_montly_hours']

data_copy[col] = MinMaxScaler().fit_transform(data_copy[col].values.reshape(-1,1)).reshape(1,-1)[0]
data_copy['salary'] = data_copy['salary'].map({'low':0, 'medium':1, 'high': 2})
data_copy['sales'] = LabelEncoder().fit_transform(data_copy['sales'].values.reshape(-1,1))
data_copy.head()
from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error
X = data_copy.drop('left', axis=1)

y = data_copy['left']
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
models = [('KNN', KNeighborsClassifier(n_neighbors=3)), 

         ('GaussianNB', GaussianNB()),

         ('LinearSVC', LinearSVC(tol=1e-5))]



for name, clf in models:

    print("%s: train score: %.3f" % (name, cross_val_score(clf, X_train, y_train, cv=5).mean()))

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("Test performance: %.3f" % (1-mean_squared_error(y_pred, y_test)))
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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelBinarizer

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.metrics import accuracy_score, plot_confusion_matrix, precision_recall_curve, plot_precision_recall_curve, average_precision_score

from sklearn.naive_bayes import BernoulliNB, GaussianNB
df = pd.read_csv("../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

df.head()
df.info()
df.plot(kind='scatter', x='sl_no', y='salary');
df.salary.describe()
df.describe()
corr = df.corr()
corr
df.columns
df.gender.value_counts()
df.ssc_b.value_counts()
df.hsc_b.value_counts()
df.hsc_s.value_counts()
df.degree_t.value_counts()
df.workex.value_counts()
df.specialisation.value_counts()
df.status.value_counts()
y = np.ravel(LabelBinarizer().fit_transform(df['status']), order='C')

X = df.drop(['sl_no', 'status'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4)
def prepare_data_pipeline(data):

    num_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary']

    cat_cols = ['ssc_b', 'hsc_b', 'hsc_s','degree_t', 'workex', 'specialisation']

    

    np = Pipeline([

        ('impute', SimpleImputer(strategy='mean')),

        ('std scaler', StandardScaler()),

    ])

    

    fp = ColumnTransformer([

        ('numer', np, num_cols),

        ('cate', OrdinalEncoder(), cat_cols),

    ])

    

    return fp.fit_transform(data)
X_prepared = prepare_data_pipeline(X_train)
model = GaussianNB()
clf = model.fit(X_prepared, y_train)
X_test_prepared = prepare_data_pipeline(X_test)
y_pred = clf.predict(X_test_prepared)
accuracy_score(y_pred, y_test)
plot_confusion_matrix(clf, X_test_prepared, y_test)
average_precision_score(y_test, y_pred)
precision_recall_curve(y_test, y_pred)
plot_precision_recall_curve(clf, X_test_prepared, y_test);
clf = BernoulliNB().fit(X_prepared, y_train)
y_pred = clf.predict(X_test_prepared)
accuracy_score(y_test, y_pred)
plot_confusion_matrix(clf, X_test_prepared, y_test)
average_precision_score(y_test, y_pred)
plot_confusion_matrix(clf, X_test_prepared, y_test)
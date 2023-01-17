# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objs as go

import plotly.offline as py

import plotly.express as px

import cv2

import seaborn

from sklearn.ensemble import RandomForestClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.options.display.float_format = '{:.2f}'.format

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score

from sklearn.metrics import plot_roc_curve

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import precision_recall_curve
df = pd.read_csv('/kaggle/input/ai4all-project/data/swab_gene_counts.csv')

df.head()
from sklearn import preprocessing

encoder = preprocessing.LabelEncoder()

df["Unnamed: 0"] = encoder.fit_transform(df["Unnamed: 0"].fillna('Nan'))

#df["Unnamed: 0"] = encoder.fit_transform(df["Unnamed: 0"].fillna('Nan'))

df.head()
# features | label 

X = df.drop('RR057e_00287', axis=1)

y = df['RR057e_00287']





# spliting 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)
from sklearn.metrics import classification_report, f1_score, confusion_matrix



def predict(model, X):

    y = model.predict(X)

    return y



def metrics(y_true, y_pred):

    print('F1 Score :', f1_score(y_true, y_pred, average='macro'))

    print(classification_report(y_true, y_pred))



    cm = confusion_matrix(y_true, y_pred)

    cm = pd.DataFrame(cm, [1,2,3,4,5], [1,2,3,4,5])



    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt="d")

    plt.show()
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, y_train)
y_train_pred = predict(clf, X_train)

metrics(y_train, y_train_pred)
y_test_pred = predict(clf, X_test)



df_submission = pd.concat([df['RR057e_00287'], pd.Series(y_test_pred, name='Unnamed: 0')], axis=1)

df_submission.to_csv('submission_MultinomialNB.csv', index=False)



df_submission
from sklearn.naive_bayes import ComplementNB

clf = ComplementNB()

clf.fit(X_train, y_train)
y_train_pred = predict(clf, X_train)

metrics(y_train, y_train_pred)
y_test_pred = predict(clf, X_test)



df_submission = pd.concat([df['RR057e_00287'], pd.Series(y_test_pred, name='Unnamed: 0')], axis=1)

df_submission.to_csv('submission_ComplementNB.csv', index=False)



df_submission
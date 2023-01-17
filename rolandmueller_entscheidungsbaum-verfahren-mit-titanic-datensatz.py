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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
url = (

    "http://biostat.mc.vanderbilt.edu/"

    "wiki/pub/Main/DataSets/titanic3.xls"

)

df = pd.read_excel(url)

orig_df = df
df
df.dtypes
df.shape
df.head()
df.describe()
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
df.groupby(['sex']).size().plot(kind='barh')

plt.show()
df.groupby(['sex', 'survived']).size().plot(kind='barh')

plt.show()
df.groupby(['sex'])['survived'].mean().plot(kind='barh')

plt.show()
df.groupby(['pclass'])['survived'].mean().plot(kind='barh')

plt.show()
df.info()
df.isnull().mean() * 100
import missingno as msno

msno.matrix(df)
msno.bar(orig_df)
msno.heatmap(df, figsize=(6, 6))
columns_to_drop = ['home.dest', 'body', 'boat', 'embarked', 'cabin', 'name', 'ticket']

df_droped = df.drop(columns= columns_to_drop)
df_droped.head(5)
df_droped.isnull().sum()
from sklearn.impute import SimpleImputer

im = SimpleImputer(strategy='median')  

df_droped[['fare', 'age']] = im.fit_transform(df_droped[['fare', 'age']])
df_droped.head()
df_droped.isnull().sum()
df_droped.info()
df_droped.head()
pd.get_dummies(df_droped)
cleaned_data = pd.get_dummies(df_droped, drop_first=True)
cleaned_data.head()
X = cleaned_data[['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']]
X
y = cleaned_data['survived']
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1, max_depth=3)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
from sklearn.tree import plot_tree

plt.figure(figsize=(20,20))

plot_tree(clf,feature_names=X_train.columns, class_names=['dead', 'survived'], filled = True)

!pip install dtreeviz
from dtreeviz.trees import dtreeviz

viz = dtreeviz(

    clf,

    X_test, 

    y_test,

    target_name="survival",

    feature_names=X_test.columns,

    class_names=['dead', 'survived'],

    scale=1.5

) 

viz
from yellowbrick.model_selection import FeatureImportances

fi_viz = FeatureImportances(clf, labels=X_test.columns)

fi_viz.fit(X_test, y_test)

fi_viz.show()
from yellowbrick.classifier import ConfusionMatrix

iris_cm = ConfusionMatrix(

    clf, classes=['dead', 'survived'],

    label_encoder={0: 'dead', 1: 'survived'}

)

iris_cm.score(X_test, y_test)

iris_cm.show()

y_predict = clf.predict(X_test)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict,target_names=['dead', 'survived']))
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_predict)
from sklearn.metrics import plot_roc_curve

svc_disp = plot_roc_curve(clf, X_test, y_test)

plt.show()
from yellowbrick.classifier import ROCAUC

fig, ax = plt.subplots(figsize=(6, 6))

roc_viz = ROCAUC(clf)

roc_viz.score(X_test, y_test)

roc_viz.show()
from yellowbrick.classifier import  PrecisionRecallCurve

viz = PrecisionRecallCurve(clf)

viz.fit(X_train, y_train)

viz.score(X_test, y_test)

viz.show()
!pip install scikit-plot

y_probas = clf.predict_proba(X_test)
import scikitplot as skplt

skplt.estimators.plot_learning_curve(clf, X, y)

plt.show()
skplt.metrics.plot_cumulative_gain(y_test, y_probas)

plt.show()
skplt.metrics.plot_lift_curve(y_test, y_probas)

plt.show()
from yellowbrick.classifier import ClassBalance

cb_viz = ClassBalance(labels=["Died", "Survived"])

cb_viz.fit(y)

cb_viz.show()
from yellowbrick.classifier import ClassPredictionError

cpe_viz = ClassPredictionError(clf, classes=["died", "survived"])

cpe_viz.score(X_test, y_test)

cpe_viz.poof()
from yellowbrick.classifier import DiscriminationThreshold

dt_viz = DiscriminationThreshold(clf)

dt_viz.fit(X, y)

dt_viz.poof()

plt.show()
from yellowbrick.classifier import ClassificationReport

mapping = {0: "died", 1: "survived"}

cm_viz = ClassificationReport(clf, classes=["died", "survived"], label_encoder=mapping)

cm_viz.score(X_test, y_test)

cm_viz.poof()

plt.show()
from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict,target_names=["died", "survived"]))
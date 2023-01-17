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
import numpy as np

import pandas as pd

from pandas import DataFrame

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas import Series,DataFrame

import scipy

from pylab import rcParams

import urllib

import sklearn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')

print ('Setup Complete')
ds=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
ds.head()
ds.describe()
ds.info()
ds_age=ds['age'].value_counts()

print(ds_age)
fig = px.histogram(ds['age'],x="age",nbins=50)

fig.show()
fig = px.histogram(ds, x="sex", color="sex", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="sex",y="DEATH_EVENT", color="sex", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="age", color="sex", marginal="rug", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="age",y="DEATH_EVENT", color="sex", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="age",y="anaemia", color="sex", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="anaemia",y="DEATH_EVENT", color="sex", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="age",y="diabetes", color="sex", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="diabetes",y="DEATH_EVENT",color ='sex', hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="age",y="smoking", color="sex", hover_data=ds.columns)

fig.show()
fig = px.histogram(ds, x="smoking",y="DEATH_EVENT", hover_data=ds.columns)

fig.show()
fig = px.scatter(ds, x="age", y="creatinine_phosphokinase", marginal_y="violin",

           marginal_x="box", template="simple_white")

fig.show()
fig = px.histogram(ds, x="creatinine_phosphokinase",y="DEATH_EVENT", hover_data=ds.columns)

fig.show()
fig = px.scatter(ds, x="age", y="ejection_fraction", marginal_y="violin",

           marginal_x="box", template="simple_white")

fig.show()
fig = px.histogram(ds, x="ejection_fraction",y="DEATH_EVENT", hover_data=ds.columns)

fig.show()
fig = px.scatter(ds, x="age", y="platelets", marginal_y="violin",

           marginal_x="box", template="simple_white")

fig.show()
fig = px.histogram(ds, x="platelets",y="DEATH_EVENT", hover_data=ds.columns)

fig.show()
fig = px.scatter(ds, x="age", y="serum_creatinine", marginal_y="violin",

           marginal_x="box", template="simple_white")

fig.show()
fig = px.histogram(ds, x="serum_creatinine",y="DEATH_EVENT", hover_data=ds.columns)

fig.show()
fig = px.scatter(ds, x="age", y="serum_sodium", marginal_y="violin",

           marginal_x="box", template="simple_white")

fig.show()
fig = px.histogram(ds, x="serum_sodium",y="DEATH_EVENT", color="sex", hover_data=ds.columns)

fig.show()
fig = px.parallel_coordinates(ds, color="DEATH_EVENT",color_continuous_scale=px.colors.diverging.Tealrose, color_continuous_midpoint=1)

fig.show()
corr = ds.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
x = ds.drop(['DEATH_EVENT'], axis=1)

y=ds.loc[:,['DEATH_EVENT']]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.15, random_state=2)
clf = RandomForestClassifier(max_depth=3, random_state=3)

clf.fit(x_train, y_train)

pred=clf.predict(x_test)

print("Accuracy of RandomForestClassifier is : ",clf.score(x_test,y_test))
titles_options = [("Confusion matrix", None)]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(clf, x_test, y_test,

                                cmap=plt.cm.Blues)

    disp.ax_.set_title(title)



    print(title)

    print(disp.confusion_matrix)



plt.show()
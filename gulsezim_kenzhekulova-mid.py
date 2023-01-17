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
import pandas as pd

test = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction-dataset/train.csv")
test.head()
train.head()
train.info()
test.info()
train.isna().sum()
test.isna().sum()
train.describe()
test.describe()
test.ID_code.value_counts()
train.shape,test.shape
import seaborn as sns

import matplotlib. pyplot as plt
sns.countplot(train.target)
sns.distplot(train.mean(axis=1),color="green", label='train')

sns.distplot(test.mean(axis=1),color="blue",label='test')
q=train.corr()

plt.figure(figsize=(15,12))

sns.heatmap(q)
test.corr()
import numpy as np

import matplotlib.pylab as plt

import plotly

import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)
X_COL = "var_81"

Y_COL = "var_68"

Z_COL = "var_108"

HUE_COL = "target"

N_SAMPLES = 10000

df = train.sample(N_SAMPLES)
# The 3D scatter plot 

trace = go.Scatter3d(

    x=df[X_COL],

    y=df[Y_COL],

    z=df[Z_COL],

    mode='markers',

    marker=dict(

        size=12,

        color=df[HUE_COL],            

        opacity=0.5,

        showscale=True,

        colorscale=[[0.0, 'black'], [1.0, 'blue']]

        

    ),

)





layout = go.Layout(

    width=600,

    height=600,

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    scene = dict(    

        xaxis = dict(

            title=X_COL),

        yaxis = dict(

            title=Y_COL),

        zaxis = dict(

            title=Z_COL),

    ),

)

fig = go.Figure(data=[trace], layout=layout)
plotly.offline.iplot(fig)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from imblearn import over_sampling

from imblearn.over_sampling import SMOTE
X = train.drop(['target', 'ID_code'],1)

y = train['target']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

Y_test = test.drop(columns = ['ID_code'])
print('X_train shape is :', X_train.shape, '\ny_train shape is ',y_train.shape)
sm = SMOTE(random_state=42)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(X_train_res, y_train_res, test_size = 0.2, random_state = 123, stratify = y_train_res)
reg=LogisticRegression()

reg.fit(X_train_res, y_train_res)
logist_pred = reg.predict_proba(X_test)[:,1]
logist_pred_test = reg.predict_proba(Y_test)[:,1]

submit = test[['ID_code']]

submit['target'] = logist_pred_test

submit.head()
submit.to_csv('logreg_baseline.csv', index = False)
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
gaus = GaussianNB()

gaus.fit(X_train,y_train)
gaus.fit(X_train,y_train)
nb_pred_test = gaus.predict_proba(Y_test)[:,1]

submit = test[['ID_code']]

submit['target'] = nb_pred_test

submit.head()
submit.to_csv('NaiveBayes.csv', index = False)
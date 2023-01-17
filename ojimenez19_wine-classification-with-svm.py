# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

import matplotlib.pyplot as plt

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

from sklearn.svm import SVC

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import plot_confusion_matrix

from sklearn.impute import SimpleImputer

from sklearn.metrics import classification_report

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')

data.head()
le = preprocessing.LabelEncoder()

data['type'] = le.fit_transform(data['type'])

data.head()
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm')
X = data.iloc[:,1:]

y=data.iloc[:,0]
fig = make_subplots(

    rows=6, cols=2,

    specs=[[{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}]],

)



plots = []



fig.add_trace(go.Histogram(x = X['fixed acidity'],name="fixed acidity"),row=1, col=1)

fig.add_trace(go.Histogram(x = X['volatile acidity'],name="volatile acidity"),row=1, col=2)



fig.add_trace(go.Histogram(x = X['citric acid'],name="citric acid"),row=2, col=1)

fig.add_trace(go.Histogram(x = X['residual sugar'],name="volatile acidity"),row=2, col=2)



fig.add_trace(go.Histogram(x = X['chlorides'],name="chlorides"),row=3, col=1)

fig.add_trace(go.Histogram(x = X['free sulfur dioxide'],name="free sulfur dioxide"),row=3, col=2)



fig.add_trace(go.Histogram(x = X['total sulfur dioxide'],name="total sulfur dioxide"),row=4, col=1)

fig.add_trace(go.Histogram(x = X['density'],name="density"),row=4, col=2)



fig.add_trace(go.Histogram(x = X['pH'],name="pH"),row=5, col=1)

fig.add_trace(go.Histogram(x = X['sulphates'],name="sulphates"),row=5, col=2)



fig.add_trace(go.Histogram(x = X['alcohol'],name="alcohol"),row=6, col=1)

fig.add_trace(go.Histogram(x = X['quality'],name="quality"),row=6, col=2)



    

fig.show()
X =pd.DataFrame(preprocessing.RobustScaler().fit_transform(X),columns=X.columns)
fig_scaled = make_subplots(

    rows=6, cols=2,

    specs=[[{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}],

           [{"type": "histogram"}, {"type": "histogram"}]],

)



plots = []



fig_scaled.add_trace(go.Histogram(x = X['fixed acidity'],name="fixed acidity"),row=1, col=1)

fig_scaled.add_trace(go.Histogram(x = X['volatile acidity'],name="volatile acidity"),row=1, col=2)



fig_scaled.add_trace(go.Histogram(x = X['citric acid'],name="citric acid"),row=2, col=1)

fig_scaled.add_trace(go.Histogram(x = X['residual sugar'],name="volatile acidity"),row=2, col=2)



fig_scaled.add_trace(go.Histogram(x = X['chlorides'],name="chlorides"),row=3, col=1)

fig_scaled.add_trace(go.Histogram(x = X['free sulfur dioxide'],name="free sulfur dioxide"),row=3, col=2)



fig_scaled.add_trace(go.Histogram(x = X['total sulfur dioxide'],name="total sulfur dioxide"),row=4, col=1)

fig_scaled.add_trace(go.Histogram(x = X['density'],name="density"),row=4, col=2)



fig_scaled.add_trace(go.Histogram(x = X['pH'],name="pH"),row=5, col=1)

fig_scaled.add_trace(go.Histogram(x = X['sulphates'],name="sulphates"),row=5, col=2)



fig_scaled.add_trace(go.Histogram(x = X['alcohol'],name="alcohol"),row=6, col=1)

fig_scaled.add_trace(go.Histogram(x = X['quality'],name="quality"),row=6, col=2)



    

fig_scaled.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.70, random_state=42)
clf = make_pipeline(SimpleImputer(strategy='mean'),SVC(gamma='auto'))

clf.fit(X_train,y_train)
scores = cross_val_score(clf, X_test, y_test, cv=5)

print(scores)

print(scores.mean())
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test)
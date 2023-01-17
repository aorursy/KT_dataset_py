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

        data = pd.read_csv(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Let's take a look at the dataset first of all

data.head()
# Make all the necessary imports



import matplotlib.pyplot as plt

!pip install lazypredict 

from lazypredict.Supervised import LazyClassifier

!pip install plotly

import plotly.express as px

import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler

from sklearn.model_selection import cross_val_score, train_test_split

from plotly.subplots import make_subplots
# Getting a outlook of our dataset

data.info()
# Checking this out to determine which are to be left as numeric and which to categorical

data.nunique()
# Separating out categorical and numerical columns





#categorical columns

cat = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

num = [l for l in list(data.columns) if l not in cat]

print("Numerical columns:",num)

print("Categorical columns:",cat)
%matplotlib inline





fig = make_subplots(rows = 2, cols = 3)



for i, n in enumerate(num):

    if n != 'target':

        fig.add_trace(go.Box(x = list(data['target']), y = list(data[n]), name = str(n)),

                     row = (i//3+1), col = i%3 + 1)



fig.show()
print('no of outliers in the thalach', sum(data['thalach']<90))

print('no of outliers in the oldpeak', sum(data['oldpeak']>5))

print('no of outliers in the chol', sum(data['chol']>400))

print('no of outliers in the trestbps', sum(data['trestbps']>190))
data = data[data['thalach']>90]

data = data[data['oldpeak']<5]

data = data[data['chol'] < 400]

data = data[data['trestbps']<190]

print("No of rows after removing the outliers", len(data))
dfcat = data[cat]

dfcat.head()
dfnum = data[num]

dfnum.head()
#One hot encoding the categorical data



onehot = OneHotEncoder()

xcat = onehot.fit_transform(dfcat.iloc[:, :].values).toarray()
# Scaling the numerical features



for col in dfnum.columns:

    dfnum[col] = (dfnum[col] - dfnum[col].mean())/dfnum[col].std()

    

xnum = dfnum.iloc[:, :-1].values



x = np.concatenate((xcat, xnum), axis = 1)

y = data.iloc[:, -1].values
# Lazypredict's Model report



xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)

models,predictions = clf.fit(xtrain, xtest, ytrain, ytest)

models
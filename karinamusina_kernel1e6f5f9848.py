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
import pandas as pd

df = pd.read_csv('/kaggle/input/flightdata (2).csv')

df.head()
df.shape
df.isna().values.any()
df.isna().sum()
df = df.drop('Unnamed: 25', axis=1)

df .isna().sum()
df = df[["MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "ORIGIN", "DEST", "CRS_DEP_TIME", "ARR_DEL15"]]

df.isnull().sum()
df[df.isna().values.any(axis=1)].head()
df = df.fillna({'ARR_DEL15': 1})

df.iloc[177:185]
df.head()
import math

for index, row in df.iterrows():

    df.loc[index, 'CRS_DEP_TIME'] = math.floor(row['CRS_DEP_TIME'] / 100)

df.head()
df = pd.get_dummies(df, columns=['ORIGIN','DEST'])

df.head()
from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(df.drop('ARR_DEL15', axis=1),

df['ARR_DEL15'], test_size=0.2, random_state=42)
train_x.shape
test_x.shape
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(random_state=13)

model.fit(train_x, train_y)
predicted = model.predict(test_x)

model.score(test_x, test_y)
from sklearn.metrics import roc_auc_score

probabilities = model.predict_proba(test_x)
roc_auc_score(test_y, probabilities[:, 1])
from sklearn.metrics import confusion_matrix

confusion_matrix(test_y, predicted)
from sklearn import metrics



disp = metrics.plot_confusion_matrix(model, test_x,

test_y, values_format = 'n',

display_labels= ["don't linger", "“delayed"])
from sklearn.metrics import precision_score
train_predictions = model.predict(train_x)

precision_score(train_y, train_predictions)
from sklearn.metrics import recall_score

recall_score(train_y, train_predictions)
from sklearn.metrics import recall_score



recall_score(train_y, train_predictions)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



sns.set()
from sklearn.metrics import roc_curve



fpr, tpr, _ = roc_curve(test_y, probabilities[:, 1])

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], color='grey', lw=1, linestyle='--')



plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

metrics.auc(fpr, tpr)

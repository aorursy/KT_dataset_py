# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt

import pandas_profiling as pp

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/voicegender/voice.csv")
df.describe()
df.info()
my_report=pp.ProfileReport(df)



my_report.to_file("my_report.html")

my_report
import missingno as miss

miss.matrix(df)

plt.show()
X=df.iloc[:, :-1]

X.head()
df.label.unique()
from sklearn.preprocessing import LabelEncoder

y=df.iloc[:,-1]



encoder = LabelEncoder()

y = encoder.fit_transform(y)

print(y)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
from sklearn.svm import SVC

from sklearn import metrics

svc=SVC()

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)
print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
from sklearn.metrics import  f1_score

f1_score = f1_score(y_test, y_pred)

print("F1 Score:")

print(f1_score)
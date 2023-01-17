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

import seaborn as sns

%matplotlib inline
data = pd.read_csv('../input/weather-dataset-rattle-package/weatherAUS.csv')

data.head()
data.columns
data.info()
data.shape
data.drop('Location',axis=1,inplace=True)

data.drop('Cloud9am',axis=1,inplace=True)

data.drop('Cloud3pm',axis=1,inplace=True)

data.drop('Sunshine',axis=1,inplace=True)

data.drop('Date',axis=1,inplace=True)

data.drop('Evaporation',axis=1,inplace=True)

data.head()
from sklearn.preprocessing import LabelEncoder

labelEncoder_df = LabelEncoder()
data['RainTomorrow']=labelEncoder_df.fit_transform(data['RainTomorrow'].astype(str))

data['WindGustDir']= labelEncoder_df.fit_transform(data['WindGustDir'].astype(str))

data['WindDir9am']= labelEncoder_df.fit_transform(data['WindDir9am'].astype(str))

data['WindDir3pm']= labelEncoder_df.fit_transform(data['WindDir3pm'].astype(str))

data['RainToday']= labelEncoder_df.fit_transform(data['RainToday'].astype(str))
data.head()
data= data.dropna(how='any')
data.isna().sum()
data.shape
from sklearn.model_selection import train_test_split
X=data.drop('RainTomorrow',axis=1)

y=data['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report

print(classification_report(y_test,predictions))

print(accuracy_score(y_test,predictions))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
pred_d=dtree.predict(X_test)
print(accuracy_score(y_test,pred_d))

print(classification_report(y_test,pred_d))
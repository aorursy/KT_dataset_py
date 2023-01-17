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

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
data=pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

data.head()
# Drop unused columns, and drop rows with any missing values.

vars = data.columns

data = data[vars].dropna()
# Check the balance of the data through plot

y=data.Outcome

ax=sns.countplot(y, label='count')

B,M=y.value_counts()

print('False',B)

print('True',M)
#separate X and y

X = data.iloc[:, :-1].values

y = data.iloc[:, 8].values
#Handling missing/zero values

from sklearn.impute import SimpleImputer as Imputer

fill_val = Imputer(missing_values = 0, strategy = 'mean')

fill_val = fill_val.fit(data.iloc[:,0:8])

X=fill_val.transform(data.iloc[:,0:8])

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



from sklearn import preprocessing

from sklearn import utils

lab_enc = preprocessing.LabelEncoder()

y_train=lab_enc.fit_transform(y_train)

y_test=lab_enc.fit_transform(y_test)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting classifier to the Training set

# Create your classifier here

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
# Predicting the Test set results

y_pred = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)



from sklearn import metrics

metrics.accuracy_score(y_test, y_pred)


y_p=pd.DataFrame(y_pred)

y_p.to_csv('out')
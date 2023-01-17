# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/red-wine-quality-cortez-et-al-2009'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data['quality'].unique()
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

%matplotlib inline
data.info()
data.isnull().sum()
import seaborn as sns

sns.set(rc={'figure.figsize':(12,9)})

sns.heatmap(data.corr(),annot=True)
sns.barplot('quality','volatile acidity',data=data)
sns.boxplot('quality','fixed acidity',data=data)
sns.barplot('quality','citric acid',data=data)
sns.barplot('quality','residual sugar',data=data)
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)
data.head()
le = LabelEncoder()

data['quality'] = le.fit_transform(data['quality'])
print(data['quality'].unique())

X = data.drop('quality',axis=1)

Y = data['quality']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state = 42,shuffle=True)
std = StandardScaler()

X_train = std.fit_transform(X_train)

X_test = std.transform(X_test)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,confusion_matrix

model = LogisticRegression()

model.fit(X_train,y_train)

prediction = model.predict(X_test)

print(classification_report(y_test,prediction))
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))
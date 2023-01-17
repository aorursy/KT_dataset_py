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

import matplotlib as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn import metrics

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

df.head(10)
df.isnull().sum()
df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

df.info()
sns.pairplot(df)
plt.pyplot.figure(figsize=(16, 16))

sns.heatmap(df.corr(), annot=True)
sns.boxplot(x='diagnosis', y='perimeter_mean', data=df)
X = df.drop('diagnosis', axis=1)

y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = svm.SVC(gamma='scale', kernel='linear')

model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(metrics.accuracy_score(prediction, y_test) * 100)
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))
df2 = pd.DataFrame(y_test)

#df1['actual']=y_test

#df1['predicted']=prediction
df2['predicted']=prediction
df2.head()
df2.to_csv('prediction.csv',index=False)
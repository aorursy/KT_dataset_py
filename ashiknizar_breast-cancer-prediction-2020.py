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
df = pd.read_csv('/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv')

df.head()
df.info()
df.describe()
import seaborn as sns
sns.pairplot(df)
sns.scatterplot(x = "mean_radius", y = "mean_area", data = df)
X = df.drop(['diagnosis'], axis=1)

X.head(2)
y = df['diagnosis']

y.head()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

fit_dtc = dtc.fit(X_train, y_train)

predict_dtc = dtc.predict(X_test)
accuracy_dtc = print(accuracy_score(predict_dtc, y_test))

matrix_dtc = print(confusion_matrix(predict_dtc, y_test))

classification_dtc = print(classification_report(predict_dtc, y_test))
rfc = RandomForestClassifier()

fit_rfc = rfc.fit(X_train, y_train)

predict_rfc = rfc.predict(X_test)
accuracy_dtc = print(accuracy_score(predict_rfc, y_test))

matrix_rfc = print(confusion_matrix(predict_rfc, y_test))

classification_rfc = print(classification_report(predict_rfc, y_test))
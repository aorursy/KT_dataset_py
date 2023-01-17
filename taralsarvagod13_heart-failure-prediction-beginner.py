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
import numpy as np
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head()
df.isna().sum()
import seaborn as sns
import matplotlib.pyplot as plt

df_corr = df.corr()
sns.heatmap(df_corr, annot=False, linewidths=0.2, center=0.5)
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X = df.drop('DEATH_EVENT', axis=1).values
y = df.iloc[:,-1]
X
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_X = pd.DataFrame(scaler.fit_transform(X))
df_X
X_train, X_test, y_train, y_test = train_test_split(df_X, y, test_size=0.2, random_state=42)
RFC = RandomForestClassifier()
RFC.fit(X_train, y_train)

y_pred = RFC.predict(X_test)
from sklearn import metrics
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
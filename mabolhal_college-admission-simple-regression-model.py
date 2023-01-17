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
ad_df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
import seaborn as sns
import matplotlib.pyplot as plt
ad_df.describe()
ad_df.drop(labels='Serial No.', axis=1, inplace=True)
sns.heatmap(ad_df.corr(), annot=True)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
X = ad_df.iloc[:, :-1]
y = ad_df.iloc[:, -1]
from sklearn.utils import shuffle
X, y = shuffle(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
ss = StandardScaler()
X_train, X_test = ss.fit_transform(X_train), ss.transform(X_test)
pd.DataFrame(X_test) # getting a glimpse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
m = LinearRegression()
m.fit(X_train, y_train)
y_pred = m.predict(X_test)
r2_score(y_test, y_pred)
pd.DataFrame(index=X.columns.values, data=m.coef_, columns = ['coeficcient'])
plt.figure(figsize=(10,15))
plt.bar(X.columns.values, m.coef_)

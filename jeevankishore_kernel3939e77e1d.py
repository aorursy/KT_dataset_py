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
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df.head()
df.shape
df.isnull().sum()
df.info()
df.columns
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X= df.drop(columns='Chance of Admit ',axis=1)
y= df['Chance of Admit ']
sdata = ss.fit_transform(X)
scaledata = pd.DataFrame(data=sdata, columns=df.columns[:7])
import seaborn as sns
sns.pairplot(data=df)
sns.heatmap(df.corr(),annot=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.ensemble import RandomForestRegressor
lr = GradientBoostingRegressor(n_estimators=200,max_depth=2,subsample=0.8,alpha=0.9)
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
lr.score(X_test,y_test)
df1 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')


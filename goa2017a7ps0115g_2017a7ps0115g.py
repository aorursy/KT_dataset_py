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
df = pd.read_csv("/kaggle/input/minor-project-2020/train.csv")
df
from sklearn.feature_selection import SelectKBest, chi2
sk_best = SelectKBest(k = 30)
import matplotlib.pyplot as plt

import seaborn as sns 

y = df['target']
X = df.drop(columns = "target")
X = sk_best.fit_transform(X, y)
import imblearn
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=125)
oversample = RandomOverSampler(sampling_strategy=0.3)
X_train, y_train = oversample.fit_resample(X_train, y_train)
from imblearn.under_sampling import RandomUnderSampler

undersample = RandomUnderSampler(sampling_strategy=0.4)

X_train, y_train = undersample.fit_resample(X_train, y_train)
X_train
from sklearn.ensemble import RandomForestClassifier
X_train.shape
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scaled_X_train = scalar.fit_transform(X_train)

scaled_X_test = scalar.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)

rf.fit(scaled_X_train, y_train)
print(rf.score(scaled_X_test, y_test))
df_test = pd.read_csv("/kaggle/input/minor-project-2020/test.csv")



X = sk_best.transform(df_test)
X_test = scalar.transform(X)
df_final = pd.DataFrame()

df_final['id'] = df_test['id']

df_final['target'] = rf.predict(X_test)
df_final

sum(df_final['target'])
df_final.to_csv("submission1.csv", index = False)
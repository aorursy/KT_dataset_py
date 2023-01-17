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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split









df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

df.tail()
df.describe()
df.info() #Check NA rows
#Change dtypes to category



catList = ["anaemia","diabetes","high_blood_pressure","sex","smoking"]



def apply_cat(df):

    for i in df.columns: #look every column name

        if i in catList: #if these columns names in our category list

            df[i] = df[i].astype("category").cat.as_ordered() #change dtype to cat

        

apply_cat(df)
df.info()
#Create model



X, y = df.drop(["DEATH_EVENT"],axis = 1), df["DEATH_EVENT"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=2)

from sklearn.ensemble import RandomForestClassifier



m = RandomForestClassifier(max_features=0.5,max_depth = 3,random_state = 1)

m.fit(X_train,y_train)

m_pred = m.predict(X_test)

print(classification_report(y_test, m_pred))
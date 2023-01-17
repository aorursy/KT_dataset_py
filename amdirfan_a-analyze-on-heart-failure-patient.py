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
#packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
df.head(5)
df.isnull().sum()
df.describe()
df['age'].hist()
plt.subplots(figsize=(10,7))
sns.heatmap(df.corr(),annot=True,linecolor='black',linewidths=3)
sns.boxplot(df['age'])
sns.boxplot(df['ejection_fraction'])
sns.countplot(x='ejection_fraction',data=df)
sns.countplot(x="ejection_fraction",hue='sex',data=df)
sns.boxplot(df['serum_sodium'])
from sklearn.model_selection import train_test_split
from  sklearn.ensemble import ExtraTreesClassifier
X = df.iloc[:,0:12]
Y = df.iloc[:,-1]
model = ExtraTreesClassifier()
model.fit(X,Y)
print(model.feature_importances_)
df.columns
X = df[['age','creatinine_phosphokinase','platelets','ejection_fraction','serum_creatinine','serum_sodium','time']]
Y = df[["DEATH_EVENT"]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier()
rc.fit(X_train,Y_train)
pred = rc.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(pred,Y_test)
print(cm)
sns.heatmap(cm,annot=True,linewidths=3,linecolor='black')
ac = accuracy_score(pred,Y_test)*100
print(ac)

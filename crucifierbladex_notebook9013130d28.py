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
df=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

df=df.drop(['id','Unnamed: 32'],axis=1)

df.head()
ports={'M':0,'B':1}

df['diagnosis']=df['diagnosis'].map(ports)

df.head()
df.info()
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

plt.figure(figsize=(12,10))

sns.heatmap(df.corr(),annot=True),df.plot()
x=df.drop(['diagnosis'],axis=1)

y=df['diagnosis'].values
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.2)

model=XGBClassifier()

x_train=scaler.fit_transform(x_train)

y_train=scaler.fit_transform(y_train)

model.fit(x_train,x_test)

y_pred=model.predict(y_train)
from sklearn import metrics 

metrics.accuracy_score(y_test,y_pred)
metrics.confusion_matrix(y_test,y_pred)
print(metrics.classification_report(y_test,y_pred))
from xgboost import plot_importance

plt.figure(figsize=(12,10))

plot_importance(model)
sns.lineplot(x=y_test,y=y_pred)

plt.legend();
y_test
y_pred
output_on_test=pd.DataFrame()

output_on_test['actual_value']=y_test

output_on_test['predicted_value']=y_pred

output_on_test.head()
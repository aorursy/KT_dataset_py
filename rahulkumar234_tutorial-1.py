# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")

train_data.head()
test_data=pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")
test_data.head()
train_data.info()
train_data['Cover_Type'].value_counts()
x=train_data.drop(["Id","Cover_Type"],axis=1)
y=train_data['Cover_Type']
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(x,y,random_state=40)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=70)

rfc.fit(x_train,y_train)
rfc.score(x_val,y_val)
predict=rfc.predict(test_data.drop("Id",axis=1))
submission=pd.DataFrame(data=predict,columns=['Cover_Type'])
submission["Id"]=test_data["Id"]

submission.set_index("Id",inplace=True)
submission.head()
submission.to_csv("Submission.csv")
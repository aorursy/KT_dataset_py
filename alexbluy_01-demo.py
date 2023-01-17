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
df_train = pd.read_csv("/kaggle/input/haitidsweek1/train.csv")

df_train.head()
df_test = pd.read_csv("/kaggle/input/haitidsweek1/test.csv")

df_test.head()
sample = pd.read_csv("/kaggle/input/haitidsweek1/sample.csv")

sample.head()
X = df_train.loc[:,df_train.columns!='label'] #feature variable

y = df_train[['label']] #target variable
X_train = X.drop('id',axis=1)
from sklearn.svm import SVC



model = SVC()



model.fit(X_train,y)
X_test = df_test.drop('id',axis=1)
y_pred = model.predict(X_test)

y_pred
y_pred = pd.DataFrame(y_pred)

y_pred.columns = ['label']

submission = pd.concat([df_test['id'],y_pred],axis=1)
submission.to_csv("submission.csv",index=False)
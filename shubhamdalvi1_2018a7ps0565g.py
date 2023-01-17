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
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')

df.head()
Y_train_raw = df['target']

X_train_raw = df.drop(['target','id'], axis = 1)

X_train_raw.head()

Y_train_raw.head()
from imblearn.over_sampling import SMOTE 

sm = SMOTE(random_state = 11) 

X_train_oversampled, Y_train_oversampled = sm.fit_sample(X_train_raw, Y_train_raw) 
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_oversampled, Y_train_oversampled)
df_test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')

df_test.head()
id_col = df_test['id']

X_test = df_test.drop(['id'], axis=1)

X_test.head()
Y_probability = model.predict_proba(X_test)

Y_probability

res=pd.DataFrame(id_col)

res['target']=pd.DataFrame(Y_probability)[1]

res
res.to_csv('submission9.csv',index=False)
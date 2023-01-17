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

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
df = pd.read_csv("/kaggle/input/costa-rican-household-poverty-prediction/train.csv")
df.head()
df.drop(['Id','idhogar','r4t3','tamhog','tamviv','hogar_total', 'SQBmeaned', 'SQBhogar_total',

            'SQBage','SQBescolari','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency',

            'SQBmeaned','agesq'], inplace = True, axis=1)
df.columns[df.isnull().sum()!=0]
df['v2a1'] = df['v2a1'].fillna((df['v2a1'].mean()))

df['v18q1'] = df['v18q1'].fillna((df['v18q1'].mean()))

df['rez_esc'] = df['rez_esc'].fillna((df['rez_esc'].mean()))

df['meaneduc'] = df['meaneduc'].fillna((df['meaneduc'].mean()))



df.columns[df.isnull().sum()!=0]
df.select_dtypes('object').head()
yes_ou_no = {'no':0,'yes':1}

df['dependency'] = df['dependency'].replace(yes_ou_no).astype(np.float32)

df['edjefe'] = df['edjefe'].replace(yes_ou_no).astype(np.float32)

df['edjefa'] = df['edjefa'].replace(yes_ou_no).astype(np.float32)
df.select_dtypes('object').head()
X = df.drop('Target', axis = 1)



y = df.Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_features=2, oob_score=True, random_state=42)

model.fit(X_train,y_train)

predicted = model.predict(X_test)

print('Classifcation report:\n', classification_report(y_test, predicted))

print('Confusion matrix:\n', confusion_matrix(y_test, predicted))
df['Target'].value_counts()
df_1 = df[df['Target']== 1]

df_2 = df[df['Target']==2]

df_3 = df[df['Target']==3]

df_4 = df[df['Target']== 4]

df_1.shape,df_2.shape,df_3.shape,df_4.shape
from sklearn.utils import resample



df_1_over = resample(df_1,

                       replace=True, # sample com replacement

                       n_samples=len(df_4), # igualando a maior classe

                       random_state=42)

df_2_over = resample(df_2,

                       replace=True, # sample com replacement

                       n_samples=len(df_4), # igualando a maior classe

                       random_state=42)

df_3_over = resample(df_3,

                       replace=True, # sample com replacement

                       n_samples=len(df_4), # igualando a maior classe

                       random_state=42)
df_over = pd.concat([df_1_over,df_2_over,df_3_over,df_4])
df_over['Target'].value_counts()

X = df_over.drop('Target', axis = 1)



y = df_over.Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,stratify=y, random_state=42)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_features=2, oob_score=True, random_state=42)

model.fit(X_train,y_train)

predicted = model.predict(X_test)

print('Classifcation report:\n', classification_report(y_test, predicted))

print('Confusion matrix:\n', confusion_matrix(y_test, predicted))
df_test = pd.read_csv("/kaggle/input/costa-rican-household-poverty-prediction/test.csv")

z = pd.Series(predicted,name="Target")

df_entrega = pd.concat([df_test.Id,z], axis=1)

df_entrega.to_csv("/kaggle/working/submission.csv",index=False)
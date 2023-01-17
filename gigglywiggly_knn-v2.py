#Jumana Nagaria
#gigglywiggly45@gmail.com
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')

#getting an insight into the data
train.head()
print(train.shape)
print(test.shape)
test.head()
test['target'] = np.nan

df = pd.concat([train, test])
df.head()
df.describe()
df.dtypes

df = df.select_dtypes(exclude=['object'])
df
#Replaces zero
zero_not_accepted=['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
for columns in zero_not_accepted:
    df[columns]=df[columns].replace(0,np.NaN)
    mean=int(df[columns].mean(skipna=True))
    df[columns]=df[columns].replace(np.NaN,mean)
df.isnull().sum()
df.target
#split dataset
X=df.iloc[:,0:8]
y=df.iloc[:,7]
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
y.shape
#feature scaling
#sc_X=StandardScaler()
#X_train=sc_X.fit_transform(df)
#X_test=sc_X.fit_transform(X_test)
import math
math.sqrt(len(df.target))

y_train.notna()
#define the model
classifier=KNeighborsClassifier(n_neighbors=221,p=2,metric='euclidean')
classifier.fit(X_train.loc[X_train['target'].notna()][zero_not_accepted], y_train[y_train.notna()])

#predict the model
y_predict=classifier.predict_proba(df.loc[df['target'].isna()][zero_not_accepted])
df_submit = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'target': y_predict[:, 1]
})
df_submit
df_submit.to_csv('submit.csv', index=False)
!head /kaggle/working/submit.csv














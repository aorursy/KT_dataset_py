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
pd.set_option('display.max_columns', 70)

pd.set_option('display.max_rows', 64)

df = pd.read_csv('/kaggle/input/deodorant-instant-liking-data/Data_train_reduced.csv')

df.head()
df.shape
df.dtypes
missing = df.isna().sum()

missing_percent = (missing/len(df)*100)

print(missing_percent)
df.drop(['q8.2','q8.8','q8.9','q8.10','q8.17','q8.18','q8.20'], axis=1, inplace=True)
df.drop(['Respondent.ID','Product'], axis=1, inplace=True)
df['q8.12'].fillna(df['q8.12'].median(), inplace=True)
df['q8.7'].fillna(df['q8.7'].median(), inplace=True)
missing = df.isna().sum()

missing_percent = (missing/len(df)*100)

print(missing_percent)
y=df['Instant.Liking']

X=df.drop('Instant.Liking', axis=1)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold



kfold = StratifiedKFold(n_splits=5,)



model = LogisticRegression(max_iter=2000)

model.fit(X,y)

result= cross_val_score(model,X,y, cv= kfold, )

print('Resultados gerais:',result)

print('Valor medio dos testes:', result.mean())
dataTest = X[5:15]
model.predict(dataTest)
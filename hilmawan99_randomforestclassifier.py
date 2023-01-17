# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/sircleai-orientation-2020/train.csv')

df.head()
df.index = df['id']

df = df.drop(columns='id')

df.head()
fac = {'fac_1':bool,'fac_2':bool,'fac_3':bool,'fac_4':bool,'fac_5':bool,'fac_6':bool,'fac_7':bool,'fac_8':bool}

df = df.astype(fac)

df.info()
y = df.gender

y.head()
feature = ['fac_1','fac_2','fac_3','fac_4','fac_5','fac_6','fac_7','fac_8','poi_1','poi_2','poi_3','size','price_monthly','room_count','total_call']

x = df[feature]

x.head()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y,random_state=100, shuffle=True)
print(len(x_train))

print(len(x_test))
1452/(1452+484)
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier()

train = clf_rf.fit(x_train, y_train)

pred = train.predict(x_test)

accuracy_score(y_test, pred)
confusion_matrix(y_test, pred)
sample_submission = pd.read_csv('/kaggle/input/sircleai-orientation-2020/sample_submission.csv')

sample_submission = sample_submission.set_index('id')

sample_submission.head()
soal = pd.read_csv('/kaggle/input/sircleai-orientation-2020/test.csv')

soal.index = soal['id']

soal = soal.drop(columns='id')

soal.head()
fac = {'fac_1':bool,'fac_2':bool,'fac_3':bool,'fac_4':bool,'fac_5':bool,'fac_6':bool,'fac_7':bool,'fac_8':bool}

soal = soal.astype(fac)

soal.info()
x_soal = soal[feature]
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

soal_scaled = scaler.fit_transform(x_soal)
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier()

train = clf_rf.fit(x_train, y_train)

jawab = train.predict(soal_scaled)
submission = sample_submission.copy()

submission['gender'] = jawab

submission.head()
submission.to_csv('submission.csv')
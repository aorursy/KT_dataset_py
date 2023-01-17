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
df = pd.read_csv("/kaggle/input/traincsv/train.csv")

print(df.head())
### null removal time

df = df[df["Kelas Pekerja"] != "?"]

df = df[df["Pendidikan"] != "?"]

df = df[df["Status Perkawinan"] != "?"]

df = df[df["Pekerjaan"] != "?"]

print(df)
print(df.describe())
def encode_gaji(s):

    if s == "<=7jt":

        return 0

    return 1

    

df["Gaji"] = df.apply(lambda row: encode_gaji(row["Gaji"]), axis=1)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
one_hot1  = pd.get_dummies(df["Pekerjaan"])

one_hot2  = pd.get_dummies(df["Jenis Kelamin"])



# X = df[["Jmlh Tahun Pendidikan","Jam per Minggu"]]

X = pd.concat([df[["Jmlh Tahun Pendidikan","Jam per Minggu"]], one_hot1, one_hot2], axis=1, sort=False)

y = df["Gaji"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
classifier = KNeighborsClassifier(n_neighbors=3, leaf_size=5, p=2)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(y_pred)

score = roc_auc_score(y_test, y_pred)

print(score)
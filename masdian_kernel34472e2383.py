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
df = pd.read_csv('../input/sanbercode-data-science-0620/train.csv')
test = pd.read_csv('../input/sanbercode-data-science-0620/test.csv')
# Peubah baru
df['x1'] = df['Umur'] * df['Jmlh Tahun Pendidikan']
df['x2'] = df['Umur'] * df['Jam per Minggu']
df['x3'] = df['Jmlh Tahun Pendidikan'] * df['Jam per Minggu']

test['x1'] = test['Umur'] * test['Jmlh Tahun Pendidikan']
test['x2'] = test['Umur'] * test['Jam per Minggu']
test['x3'] = test['Jmlh Tahun Pendidikan'] * test['Jam per Minggu']
# mengganti '?' dengan modus (Imputasi)
import numpy as np

df['Kelas Pekerja'] = df['Kelas Pekerja'].replace(to_replace='?', value=np.nan)
df['Kelas Pekerja'] = df['Kelas Pekerja'].fillna(df['Kelas Pekerja'].mode())

df['Pekerjaan'] = df['Pekerjaan'].replace(to_replace='?', value=np.nan)
df['Pekerjaan'] = df['Pekerjaan'].fillna(df['Pekerjaan'].mode())

test['Kelas Pekerja'] = test['Kelas Pekerja'].replace(to_replace='?', value=np.nan)
test['Kelas Pekerja'] = test['Kelas Pekerja'].fillna(df['Kelas Pekerja'].mode())

test['Pekerjaan'] = test['Pekerjaan'].replace(to_replace='?', value=np.nan)
test['Pekerjaan'] = test['Pekerjaan'].fillna(df['Pekerjaan'].mode())
# Membuang peubah 'Pendidikan' karena sama dengan 'Jmlh tahun Pendidikan' dan juga id
df = df.drop('id', axis = 1)

test1 = test.drop('id', axis = 1)
# Mengubah 'Keuntungan Kapital' jadi 0 dan 1
df.loc[df['Keuntungan Kapital'] == 0.0, 'Keuntungan Kapital'] = 'Tidak Untung'
df.loc[df['Keuntungan Kapital'] != '0.0', 'Keuntungan Kapital'] = 'Untung'
#df['Keuntungan Kapital'] = df['Keuntungan Kapital'].astype('int64')

# Mengubah 'Kerugian Capital' jadi 0 dan 1
df.loc[df['Kerugian Capital'] == 0.0, 'Kerugian Capital'] = 'Tidak Rugi'
df.loc[df['Kerugian Capital'] != '0.0', 'Kerugian Capital'] = 'Rugi'
#df['Kerugian Capital'] = df['Kerugian Capital'].astype('int64')

# Mengubah 'Keuntungan Kapital' jadi 0 dan 1
test1.loc[df['Keuntungan Kapital'] == 0.0, 'Keuntungan Kapital'] = 'Tidak Untung'
test1.loc[df['Keuntungan Kapital'] != '0.0', 'Keuntungan Kapital'] = 'Untung'
#test1['Keuntungan Kapital'] = test1['Keuntungan Kapital'].astype('int64')

# Mengubah 'Kerugian Capital' jadi 0 dan 1
test1.loc[df['Kerugian Capital'] == 0.0, 'Kerugian Capital'] = 'Tidak Rugi'
test1.loc[df['Kerugian Capital'] != '0.0', 'Kerugian Capital'] = 'Rugi'
#test1['Kerugian Capital'] = test1['Kerugian Capital'].astype('int64')
# Standarisasi

from sklearn.preprocessing import StandardScaler

stdscalar = StandardScaler()

norm_var = ['Umur', 'Berat Akhir', 'Jmlh Tahun Pendidikan', 'Jam per Minggu', 'x1', 'x2', 'x3']

df[norm_var] = stdscalar.fit_transform(df[norm_var])
test1[norm_var] = stdscalar.transform(test1[norm_var])
# Encoding data nominal
df = pd.get_dummies(df, columns = ['Kelas Pekerja', 'Pendidikan', 'Status Perkawinan', 'Pekerjaan', 'Jenis Kelamin', 'Keuntungan Kapital', 'Kerugian Capital'])

test1 = pd.get_dummies(test1, columns = ['Kelas Pekerja', 'Pendidikan', 'Status Perkawinan', 'Pekerjaan', 'Jenis Kelamin', 'Keuntungan Kapital', 'Kerugian Capital'])
# Mengubah variable target menjadi 0 dan 1
df.loc[df['Gaji'] == '<=7jt', 'Gaji'] = 0
df.loc[df['Gaji'] == '>7jt', 'Gaji'] = 1
df['Gaji'] = df['Gaji'].astype('int64')
# memisahkan feature dan target pada data train
x = df.drop('Gaji', axis = 1)
y = df['Gaji']
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# Membuat objek model dan memilih hyperparameter
mod_knn = KNeighborsClassifier() 
mod_lr = LogisticRegression(random_state=21)
mod_rf = RandomForestClassifier(random_state=21)
mod_dt = tree.DecisionTreeClassifier(random_state=21)
# KNN
cv = cross_val_score(mod_knn, x, y, cv = 5, scoring='roc_auc')
cv.mean()
# Logistic Regression
cv = cross_val_score(mod_lr, x, y, cv = 5, scoring='roc_auc')
cv.mean()
# Random Forest
cv = cross_val_score(mod_rf, x, y, cv = 5, scoring='roc_auc')
cv.mean()
# Decisio tree
cv = cross_val_score(mod_dt, x, y, cv = 5, scoring='roc_auc')
cv.mean()
# Logistik paling bagus. lanjut ke tuning hyper parameter
mod_lr = LogisticRegression(random_state=21)
params_grid = {
    'C':np.arange(0.1, 1, 0.1), 'class_weight': [{0:x, 1:1-x} for x in np.arange(0.1, 0.9, 0.1)]
}
rscv = RandomizedSearchCV(mod_lr, params_grid, cv = 5, scoring = 'roc_auc', n_iter = 20)
rscv.fit(x, y)
rscv.best_score_
# prediksi test.csv
y_pred = rscv.predict(test1)
# Menyimpan hsil untuk submission
hasil = {'id' : test['id'], 'Gaji' : y_pred}
hasil = pd.DataFrame(hasil)
hasil.to_csv('hasil.csv', index = False)
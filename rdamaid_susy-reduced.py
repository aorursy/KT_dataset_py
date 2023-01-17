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
# Impor data
data = pd.read_csv("../input/susyreduced/SUSYReduced.csv")
data.head()
# Bagi data menjadi data train dan data test
columns = "lepton1eta lepton1phi lepton2pT lepton 2eta lepton2phi missingenergymagnitude missingenergyphi MET_rel axialMET M_R M_TR_2 R MT2 S_R M_Delta_R dPhi_R_b cos(theta_r1)".split() # Declare the columns names
df = pd.DataFrame(data, columns=columns)
colY = "class".split()
y = pd.DataFrame(data, columns=colY)

df.describe()
df.isnull().sum()
# Karena kolom lepton dan 2eta mengandung missing value semua
# Menghilangkan kolom lepton dan 2eta
df.drop(['lepton', '2eta'], axis = 1, inplace = True) 
df.isnull().sum()
# Perlu praproses normalisasi
from sklearn import preprocessing
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
norm_df = pd.DataFrame(x_scaled)

norm_df.describe()
y.isnull().sum()
y.describe()
# create training and testing vars
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

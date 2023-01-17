!pip install lifelines
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
import sklearn

import matplotlib.pyplot as plt



from lifelines import CoxPHFitter

from lifelines.utils import concordance_index as cindex

from sklearn.model_selection import train_test_split
df = pd.read_csv('/kaggle/input/mayo-clinic-primary-biliary-cirrhosis-data/pbc.csv')
df.head()
for i in df.index:

    df.at[i, 'sex'] = 0 if df.loc[i,'sex'] == "f" else 1
df.head()
np.random.seed(0)

df_dev, df_test = train_test_split(df, test_size = 0.2)

df_train, df_val = train_test_split(df_dev, test_size = 0.25)
continuous_columns = ['age', 'bili', 'chol', 'albumin', 'copper', 'alk.phos', 'ast', 'trig', 'platelet', 'protime']

mean = df_train.loc[:, continuous_columns].mean()

std = df_train.loc[:, continuous_columns].std()

df_train.loc[:, continuous_columns] = (df_train.loc[:, continuous_columns] - mean) / std

df_val.loc[:, continuous_columns] = (df_val.loc[:, continuous_columns] - mean) / std

df_test.loc[:, continuous_columns] = (df_test.loc[:, continuous_columns] - mean) / std
df_train.loc[:, continuous_columns].describe()
def one_hot_encoder(dataframe, columns):

    return pd.get_dummies(dataframe, columns = columns, drop_first = True, dtype=np.float64)
to_encode = ["edema", "stage"]



one_hot_train = one_hot_encoder(df_train, to_encode)

one_hot_val = one_hot_encoder(df_val, to_encode)

one_hot_test = one_hot_encoder(df_test, to_encode)



print(one_hot_val.columns.tolist())

print(f"There are {len(one_hot_val.columns)} columns")
one_hot_train.head()
one_hot_train.dropna(inplace=True)
cph = CoxPHFitter()

cph.fit(one_hot_train, duration_col = 'time', event_col = 'status', step_size=0.1)
cph.print_summary()
cph.plot_covariate_groups('trt', values=[0, 1]);
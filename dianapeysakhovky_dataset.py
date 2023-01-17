!ls /kagle 
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
#функция для сортировки таблицы. У каждого субъекта используем электроды  Fz, FCz, Cz, FC3, FC4, C3, C4, CP3, CP4. Это

df1 = pd.read_csv('/kaggle/input/button-tone-sz/1.csv/1.csv')

df1
pd.read_csv('/kaggle/input/button-tone-sz/1.csv/1.csv')
column_names_df = pd.read_csv("/kaggle/input/button-tone-sz/columnLabels.csv")
column_names_df.columns
#сколько в датафрейме строк и колонок. 
df1.shape
#колонки начальной таблицы пациента
df1.columns
#отдельная переменная для соотвествующего названия колонок
column_names_df.columns

#column_names_df.columns['subject', 'trial', 'condition', 'sample' , 'Fz', 'FCz', 'Cz', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4']

df1 = pd.read_csv("/kaggle/input/button-tone-sz/1.csv/1.csv", header=None, names=column_names_df.columns)
df1

df_main = df1.copy()
a = df_main[['subject', 'trial', 'condition' , 'Fz', 'FCz', 'Cz', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4']]
a
a[1:10]
b = df_main[['subject' , 'trial' ,'condition' , 'Fz', 'FCz', 'Cz', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4']]
b[ b ['trial'] == 100]

 !pip install xgboost

    

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
# load data
dataset = a

# split data into X and y
X = dataset[:,8]
Y = dataset[:,8]

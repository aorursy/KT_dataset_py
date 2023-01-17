# Імпорт Pandas і Numpy

import numpy as np

import pandas as pd



#Вхідні дані доступні в "../input/"

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Будь-які результати доступні у папці "../output/kaggle/working"
df = pd.read_csv("../input/telecom-churn/telecom_churn.csv")

df.head(2)
df.shape
df.columns
df.info()
df['churn'] = df['churn'].astype('int64')
df.info()
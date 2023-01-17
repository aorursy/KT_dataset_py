import numpy as np

import pandas as pd



df = pd.read_csv('/kaggle/input/traincsv/train-200907-141856.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include='object')
df['Cabin'].value_counts()
df['Cabin'].value_counts(normalize='True')
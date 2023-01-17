import pandas as pd

import numpy as np



df = pd.read_csv('../input/machine-learning-on-titanic-data-set/train.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include='object')
df['Embarked'].value_counts()
df['Sex'].value_counts(normalize='True')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('../input/edataanalysiscsv/train.csv')
df.head()
print(df.shape)
df.columns
print(df.info())
df.describe()
df.describe(include='object')
df['Sex'].value_counts()
df['Embarked'].value_counts()
df['Sex'].value_counts(normalize = True)
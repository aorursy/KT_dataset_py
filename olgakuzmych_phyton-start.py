import os
os.listdir('../input/breast-cancer-wisconsin-data')
import pandas as pd
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head(33)
df.shape
df.columns
df.describe()
column_array = ['radius_se', 'texture_se','concavity_worst', 'concave points_worst']
df[column_array]. describe()
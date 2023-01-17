import os
os.listdir('../input/breast-cancer-wisconsin-data')
import pandas as pd
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head()
df.shape
df.columns
df.describe()
column_array=['compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se']
df[column_array]. describe()
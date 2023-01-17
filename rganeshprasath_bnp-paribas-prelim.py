import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier as RC
from sklearn.decomposition import PCA
import csv
df = pd.read_csv('../input/train.csv',header = 0)
test_df = pd.read_csv('../input/test.csv')
for i in df.columns:
    print (df[i].dtype)
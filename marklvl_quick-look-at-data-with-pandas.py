import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_df = pd.read_csv('../input/train.csv', index_col='Id')
train_df.head()
train_df.info()
train_df.describe()
train_df.MSZoning.value_counts()
import matplotlib.pyplot as plt
train_df.hist(bins=20, figsize=(30,30))
plt.show()

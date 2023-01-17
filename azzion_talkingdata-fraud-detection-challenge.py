import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train_sample.csv')
df.head(5)
df.isnull().sum()
df.iloc[np.where(df['is_attributed'].values==0)]
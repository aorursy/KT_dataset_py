import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualization librariy
df = pd.read_csv('../input/data_set_ALL_AML_train.csv')
df.describe()
x = df['1']

sns.distplot(x,kde= False)
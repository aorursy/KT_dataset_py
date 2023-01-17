import pandas as pd

import seaborn as sns



cereals = pd.read_csv('../input/cereal.csv')
cereals.describe()
cereals.dtypes
sns.distplot(cereals['calories'], kde=False).set_title('Cereals\' calories distribution')
sns.distplot(cereals['calories'], bins=10).set_title("Cereals' calories distribution")
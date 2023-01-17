import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
df = pd.read_csv('../input/cereal.csv')
df.head()
df_category = df['mfr']
g = sns.countplot(x=df_category)
g.set_title('Mfr types')

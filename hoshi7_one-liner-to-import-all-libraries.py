!pip install pyforest

from pyforest import *
active_imports()
df = pd.DataFrame(pd.read_csv('../input/train.csv'))
active_imports()
df['Age'].dropna(inplace= True)

sns.distplot(df['Age'])
df.profile_report()
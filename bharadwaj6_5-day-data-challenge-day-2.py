import numpy as np

import pandas as pd

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
pokemon = pd.read_csv('../input/pokemon.csv')
pokemon.head()
pokemon.info()
pokemon.describe()
sns.distplot(pokemon['Speed'], kde=False)
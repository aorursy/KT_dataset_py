import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats
cereals = pd.read_csv('../input/cereal.csv')

hot = cereals.loc[cereals['type'] == 'H']

cold = cereals.loc[cereals['type'] == 'C']

f_hot = hot['mfr'].value_counts()

f_cold = cold['mfr'].value_counts()
scipy.stats.chisquare(f_hot)
scipy.stats.chisquare(f_cold)
contingency = pd.crosstab(cereals['mfr'], cereals['type'])

contingency
scipy.stats.chi2_contingency(contingency)
import seaborn as sns



sns.countplot(x='mfr', hue='type', data=cereals)
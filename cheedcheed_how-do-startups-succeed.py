import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/crunch2013"))

# Any results you write to the current directory are saved as output.
# Loading data 

df = pd.read_csv("../input/crunch2013/crunchbase-investments.csv", encoding = "ISO-8859-1")
df.sort_values(['company_permalink', 'funded_month'], inplace=True)
df.head()
# Group by the investment round

grp = df.groupby('funding_round_type')
grp['raised_amount_usd'].describe()
f, ax = plt.subplots(figsize=(16, 16))
ax.set_xscale("log")

sns.set(style="ticks")
sns.boxplot(y='funding_round_type', x='raised_amount_usd', 
            data=df, palette="vlag", whis=np.inf)

ax.set(ylabel="", xlabel="Investments in Log-Dollar")
ax.xaxis.grid(True)
sns.despine(trim=True, left=True)

end = 'zend'

# Round Order can be derived from d.sort_values('mean').index
round_order = ['angel', 'series-a', 'series-b', 'series-c+', end]
# ALT: 'crowdfunding', 'venture', 'other', 'private-equity', 'post-ipo'

m = {r: {k: 0 for k in round_order} for r in round_order}

rounds = pd.read_csv("../input/crunch2013/crunchbase-rounds.csv", encoding = "ISO-8859-1")
rounds.sort_values(['company_permalink', 'funded_month'], inplace=True)

companies = rounds.groupby('company_permalink')
for _, group in companies:
    z = np.append(group['funding_round_type'].values, [end])
    
    for src, dest in zip(z, np.roll(z, -1)):
        try:
            m[src][dest] += 1
        except KeyError:
            pass

del m[end]
g = pd.DataFrame(m).T

f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(g, annot=True, fmt='d')
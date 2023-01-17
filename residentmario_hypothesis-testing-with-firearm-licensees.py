import pandas as pd

import numpy as np

licensees = pd.read_csv("../input/federal-firearm-licensees.csv", index_col=0)[1:]

licensees.head(3)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

licensees['Premise Zip Code'].value_counts().plot.hist(bins=50)
licensees['Premise Zip Code'].value_counts().mean()
X = licensees['Premise Zip Code'].value_counts()
import numpy as np

import scipy.stats as stats



def t_value(X, h_0):

    se = np.sqrt(np.var(X) / len(X))

    return (np.mean(X) - h_0) / se



def p_value(t):

    # Two-sided p-value, so we multiply by 2.

    return stats.norm.sf(abs(t))*2



t = t_value(X, 2.75)

p = p_value(t)
t, p
import scipy.stats as stats



stats.ttest_1samp(a=X, popmean=2.75)
r = (licensees['Premise Zip Code']

         .value_counts()

         .sample(len(licensees['Premise Zip Code'].unique()) - 1))

pd.Series(r.cumsum() / np.array(range(1, len(r) + 1))).reset_index(drop=True).plot.line(

    figsize=(12, 4), linewidth=1

)
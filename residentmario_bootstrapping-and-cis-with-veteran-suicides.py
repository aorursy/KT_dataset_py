import pandas as pd

import numpy as np

veterans_2005 = pd.read_csv("../input/2005.csv", index_col=0)

veterans_2005.head(3)
df_2005 = pd.DataFrame(

    {'vet': veterans_2005['vet_suicides'] / veterans_2005['vet_pop'],

     'civ': (veterans_2005['all_suicides'] - veterans_2005['vet_suicides']) / 

            (veterans_2005['overall_pop_18'] - veterans_2005['vet_pop'])}

)

df_2005.head(3)
import scipy.stats as st



def confidence_interval(X, c):

    x_bar = np.mean(X)

    z_score = st.norm.ppf(1 - ((1 - c) / 2))

    sqrt_n = np.sqrt(len(X))

    std_dev = np.std(X)

    

    delta = z_score * (std_dev / sqrt_n)

    return np.array([x_bar - delta, x_bar + delta])
confidence_interval(df_2005.civ, 0.95)
confidence_interval(df_2005.vet, 0.95)
confidence_interval(df_2005.civ, 0.95) * 1000000
confidence_interval(df_2005.vet, 0.95) * 1000000
draws = np.array([np.random.choice(df_2005.civ, size=20) for _ in range(10000)]) * 1000000

civ_means = np.array([np.mean(draw) for draw in draws])



draws = np.array([np.random.choice(df_2005.vet, size=20) for _ in range(10000)]) * 1000000

vet_means = np.array([np.mean(draw) for draw in draws])



del draws
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

plt.suptitle("Suicides Per Million Estimators, Bootstrapped").set_position([.5, 1.05])



import seaborn as sns

sns.distplot(pd.Series(civ_means), ax=axarr[0])

axarr[0].set_title("Civilians")



sns.distplot(pd.Series(vet_means), ax=axarr[1])

axarr[1].set_title("Veterans")

pass
years = range(2005, 2012)



cis = []

for year in years:

    df = pd.read_csv("../input/{0}.csv".format(year), index_col=0)

    df = pd.DataFrame(

        {'vet': df['vet_suicides'] / df['vet_pop'],

         'civ': (df['all_suicides'] - df['vet_suicides']) / 

                (df['overall_pop_18'] - df['vet_pop'])}

    )

    cis.append({'civ': confidence_interval(df.civ, 0.95),

                'vet': confidence_interval(df.vet, 0.95)})
civ_means = [np.mean(c['civ'])*10**6 for c in cis]

vet_means = [np.mean(c['vet'])*10**6 for c in cis]



civ_mins = [c['civ'][0]*10**6 for c in cis]

vet_mins = [c['vet'][0]*10**6 for c in cis]



civ_maxs = [c['civ'][1]*10**6 for c in cis]

vet_maxs = [c['vet'][1]*10**6 for c in cis]
ind = pd.Index(range(2005, 2012))



fig, axarr = plt.subplots(1, 2, figsize=(12, 4))

plt.suptitle("Suicides Per Million Estimates, 2005-2011").set_position([.5, 1.05])



pd.Series(civ_means, index=ind).plot.line(color='black', ax=axarr[0])

pd.Series(civ_mins, index=ind).plot.line(color='steelblue', ax=axarr[0])

pd.Series(civ_maxs, index=ind).plot.line(color='steelblue', ax=axarr[0])

axarr[0].set_title("Civilians")



pd.Series(vet_means, index=ind).plot.line(color='black', ax=axarr[1])

pd.Series(vet_mins, index=ind).plot.line(color='steelblue', ax=axarr[1])

pd.Series(vet_maxs, index=ind).plot.line(color='steelblue', ax=axarr[1])

axarr[1].set_title("Veterans")

pass
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib

import statsmodels.api as sm



matplotlib.style.use('fivethirtyeight')





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



game = pd.read_csv('../input/summer.csv')

country = pd.read_csv('../input/dictionary.csv')
import warnings 



warnings.filterwarnings('ignore')





game['gold']=0

game['silver']=0

game['bronze']=0

game['gold'][game['Medal']=='Gold'] = 1

game['silver'][game['Medal']=='Silver'] = 1

game['bronze'][game['Medal']=='Bronze'] = 1
gsb = game.groupby(['Country']).sum()[['gold', 'silver', 'bronze']]

plt.plot(gsb['gold'], gsb['silver'], 'ro', label = '$Silver$')

plt.plot(gsb['gold'], gsb['bronze'], 'gs', label = '$Bronze$')

plt.xscale('log'); plt.yscale('log')

plt.xlabel(r'$Gold$', fontsize = 20)

plt.ylabel(r'$Medal$', fontsize = 20)

plt.legend(loc = 2, numpoints = 1, fontsize = 20, frameon = False)

plt.show()
game['score']=0

game['score'][game['Medal']=='Gold'] = 4

game['score'][game['Medal']=='Silver'] = 2

game['score'][game['Medal']=='Bronze'] = 1
def gini_coefficient(v):

    bins = np.linspace(0., 100., 11)

    total = float(np.sum(v))

    yvals = []

    for b in bins:

        bin_vals = v[v <= np.percentile(v, b)]

        bin_fraction = (np.sum(bin_vals) / total) * 100.0

        yvals.append(bin_fraction)

    # perfect equality area

    pe_area = np.trapz(bins, x=bins)

    # lorenz area

    lorenz_area = np.trapz(yvals, x=bins)

    gini_val = (pe_area - lorenz_area) / float(pe_area)

    return bins, yvals, gini_val
score_all = game.groupby(['Country']).sum()['score']

bins, result, gini_val = gini_coefficient(score_all)



plt.plot(bins, result, label="observed")

plt.plot(bins, bins, '--', label="perfect eq.")

plt.xlabel("fraction of books")

plt.ylabel("fraction of degree centrality")

plt.title("GINI: %.4f" %(gini_val))

plt.legend(loc=0)

plt.show()
game['award']=1

disciplines = game.Discipline.unique()

gg = game.groupby(['Discipline', 'Country']).sum()

ggds = gg['score']['Swimming']

ggds_max = ggds.sort_values(ascending = False).iloc[0]

gg_sum = np.sum(ggds)

gg_max  = ggds.sort_values(ascending = False)

gg_max_value, gg_max_index = gg_max.iloc[0], gg_max.index[0]

gg_max_ratio = np.float(gg_max_value)/gg_sum

print(gg_max_value, gg_max_ratio, gg_max_index)
for i in disciplines:

    if len(gg['award'][i]) > 10:

        print(i, gini_coefficient(gg['award'][i])[2])
for i in disciplines:

    if len(gg['score'][i]) > 10:

        print(i, gini_coefficient(gg['score'][i])[2])
medal_score = []

for i in country.Code:

    if i in score_all.index:

        medal_score.append(score_all[i])

    else:

        medal_score.append(0)

        

country['medal_score'] = medal_score



matplotlib.style.use('fivethirtyeight')



plt.plot(country['Population'], country['medal_score'], 'ro')

plt.xscale('log'); plt.yscale('log')

plt.xlabel(r'$Population$', fontsize = 20)

plt.ylabel(r'$Medal\; Score$', fontsize = 20)

plt.show()
plt.plot(country['GDP per Capita'], country['medal_score'], 'gs')

plt.xscale('log'); plt.yscale('log')

plt.xlabel(r'$GDP \;per\; Capita$', fontsize = 20)

plt.ylabel(r'$Medal\; Score$', fontsize = 20)

plt.show()
years = game.Year.unique()

ggy = game.groupby(['Year', 'Country']).sum()['score']

gini = [gini_coefficient(ggy[i])[2] for i in years]



fig = plt.figure(figsize=(12, 4),facecolor='white')



plt.plot(years, gini, 'r-o')

plt.ylabel(r'$Gini\; Coefficients$', fontsize = 20)

plt.show()

gg = game.groupby(['Discipline', 'Country']).sum()

for i in disciplines:

    ggds = gg['score'][i]

    gg_sum = np.sum(ggds)

    gg_max  = ggds.sort_values(ascending = False)

    gg_max_value, gg_max_index = gg_max.iloc[0], gg_max.index[0]

    gg_max_ratio = np.float(gg_max_value)/gg_sum

    if gg_max_ratio >= .5:

        print (i, gg_max_ratio, gg_max_index)
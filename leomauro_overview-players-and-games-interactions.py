# Pandas

import numpy as np

import pandas as pd

# Plot

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt



mpl.style.use('seaborn-whitegrid')

%matplotlib inline
dataset_folder = '../input'

plot_folder = '../plot'



courses = pd.read_csv('%s/%s' % (dataset_folder, 'courses.csv'), sep='\t', encoding='utf-8')

likes   = pd.read_csv('%s/%s' % (dataset_folder, 'likes.csv'), sep='\t', encoding='utf-8')

plays   = pd.read_csv('%s/%s' % (dataset_folder, 'plays.csv'), sep='\t', encoding='utf-8')

clears  = pd.read_csv('%s/%s' % (dataset_folder, 'clears.csv'), sep='\t', encoding='utf-8')

records = pd.read_csv('%s/%s' % (dataset_folder, 'records.csv'), sep='\t', encoding='utf-8')
ids = courses['id'].unique().tolist()

interactions = {id:{'likes':0, 'plays':0, 'clears':0, 'records':0} for id in ids}
names = ['courses','likes','plays','clears','records']

for df, name in zip([courses,likes,plays,clears,records], names):

    print('%s:' % (name), len(df))
likes.head()
# count number of likes per map

likes_per_course = likes['id'].value_counts().to_dict()
for id, values in likes_per_course.items():

    interactions[id]['likes'] = values
plays.head()
# count number of plays per map

plays_per_course = plays['id'].value_counts().to_dict()
for id, values in plays_per_course.items():

    interactions[id]['plays'] = values
clears.head()
# count number of clears per map

clears_per_course = clears['id'].value_counts().to_dict()
for id, values in clears_per_course.items():

    interactions[id]['clears'] = values
records.head()
# count number of records per map

records_per_course = records['id'].value_counts().to_dict()
for id, values in records_per_course.items():

    interactions[id]['records'] = values
# palette of colors

palette = sns.color_palette('cubehelix', 4)

sns.palplot(palette)
# data sorted

df = pd.DataFrame(interactions).transpose()

df['sum'] = df['likes'] + df['plays'] + df['clears'] + df['records']

df = df.sort_values(by=['sum'], ascending=False)
df.head()
# settings

limit = 100

fontsize = 14



# getting axis

axis_id = df.index.tolist()[0:limit]

axis_plays = df['plays'].tolist()[0:limit]

axis_clears = df['clears'].tolist()[0:limit]

axis_records = df['records'].tolist()[0:limit]

axis_likes = df['likes'].tolist()[0:limit]
# plot

fig, ax = plt.subplots()

bottom_records  = [axis_plays[i] + axis_clears[i] for i in range(0, limit)]

bottom_likes    = [bottom_records[i] + axis_records[i] for i in range(0, limit)]



# bar plot

p1 = plt.bar(range(0, limit), axis_plays, color=palette[0], label='Plays')

p2 = plt.bar(range(0, limit), axis_clears, bottom=axis_plays, color=palette[1], label='Clears')

p3 = plt.bar(range(0, limit), axis_records, bottom=bottom_records, color=palette[2], label='Records')

p4 = plt.bar(range(0, limit), axis_likes, bottom=bottom_likes, color=palette[3], label='Likes')



# texts and labels

plt.ylabel('Players Interactions', fontsize=fontsize)

plt.xlabel('Game Maps', fontsize=fontsize)

ax.legend(prop={'size':fontsize-2})



# ticks

fig.set_size_inches(6, 3, forward=True)

plt.xlim(-1, 100)

# plt.savefig('%s/%s.pdf' % (plot_folder, 'interactions'), dpi=300)

plt.show()
!pip install powerlaw
import powerlaw

from scipy import stats

# -- ignore warning

import warnings

warnings.simplefilter(action='ignore', category=RuntimeWarning)
data = df['sum']

fit = powerlaw.Fit(data, discrete=True, estimate_discrete=False)
# plot

fig, ax = plt.subplots()

fig_powerlaw = fit.plot_pdf(linewidth=3, color=palette[0], label='Empirical data')

fit.power_law.plot_pdf(ax=fig_powerlaw, color=palette[1], linestyle='--', label='Power law fit')

fit.lognormal.plot_pdf(ax=fig_powerlaw, color=palette[2], linestyle='--', label='Log-normal fit')



# texts and labels

ax.legend(prop={'size':fontsize-2})



# ticks

fig.set_size_inches(6, 3, forward=True)

# plt.savefig('%s/%s.pdf' % (plot_folder, 'interactions-powerlaw-PDF'), dpi=300)

plt.show()
# plot

fig, ax = plt.subplots()

fig_powerlaw = fit.plot_cdf(linewidth=3, color=palette[0], label='Empirical data')

fit.power_law.plot_cdf(ax=fig_powerlaw, color=palette[1], linestyle='--', label='Power law fit')

fit.lognormal.plot_cdf(ax=fig_powerlaw, color=palette[2], linestyle='--', label='Log-normal fit')



# texts and labels

ax.legend(prop={'size':fontsize-2})



# ticks

fig.set_size_inches(6, 3, forward=True)

# plt.savefig('%s/%s.pdf' % (plot_folder, 'interactions-powerlaw-CDF'), dpi=300)

plt.show()
# plot

fig, ax = plt.subplots()

fig_powerlaw = fit.plot_ccdf(linewidth=3, color=palette[0], label='Empirical data')

fit.power_law.plot_ccdf(ax=fig_powerlaw, color=palette[1], linestyle='--', label='Power law fit')

fit.lognormal.plot_ccdf(ax=fig_powerlaw, color=palette[2], linestyle='--', label='Log-normal fit')



# texts and labels

ax.legend(prop={'size':fontsize-2})



# ticks

fig.set_size_inches(6, 3, forward=True)

# plt.savefig('%s/%s.pdf' % (plot_folder, 'interactions-powerlaw-CCDF'), dpi=300)

plt.show()
cdfs = [

    "norm",            #Normal (Gaussian)

    "alpha",           #Alpha

    "anglit",          #Anglit

    "arcsine",         #Arcsine

    "beta",            #Beta

    "betaprime",       #Beta Prime

    "bradford",        #Bradford

    "burr",            #Burr

    "cauchy",          #Cauchy

    "chi",             #Chi

    "chi2",            #Chi-squared

    "cosine",          #Cosine

    "dgamma",          #Double Gamma

    "dweibull",        #Double Weibull

    "erlang",          #Erlang

    "expon",           #Exponential

    "exponweib",       #Exponentiated Weibull

    "exponpow",        #Exponential Power

    "fatiguelife",     #Fatigue Life (Birnbaum-Sanders)

    "foldcauchy",      #Folded Cauchy

    "f",               #F (Snecdor F)

    "fisk",            #Fisk

    "foldnorm",        #Folded Normal

    "gamma",           #Gamma

    

#     "gausshyper",      #Gauss Hypergeometric

    

    "genexpon",        #Generalized Exponential

    "genextreme",      #Generalized Extreme Value

    "gengamma",        #Generalized gamma

    "genlogistic",     #Generalized Logistic

    "genpareto",       #Generalized Pareto

    "genhalflogistic", #Generalized Half Logistic

    "gilbrat",         #Gilbrat

    "gompertz",        #Gompertz (Truncated Gumbel)

    "gumbel_l",        #Left Sided Gumbel, etc.

    "gumbel_r",        #Right Sided Gumbel

    "halfcauchy",      #Half Cauchy

    "halflogistic",    #Half Logistic

    "halfnorm",        #Half Normal

    "hypsecant",       #Hyperbolic Secant

    "invgamma",        #Inverse Gamma

    "invweibull",      #Inverse Weibull

    "johnsonsb",       #Johnson SB

    "johnsonsu",       #Johnson SU

    "laplace",         #Laplace

    "logistic",        #Logistic

    "loggamma",        #Log-Gamma

    "loglaplace",      #Log-Laplace (Log Double Exponential)

    "lognorm",         #Log-Normal

    "lomax",           #Lomax (Pareto of the second kind)

    "maxwell",         #Maxwell

    "mielke",          #Mielke's Beta-Kappa

    "nakagami",        #Nakagami

    

#     "ncx2",            #Non-central chi-squared

#     "ncf",             #Non-central F

#     "nct",             #Non-central Student's T

    

    "pareto",          #Pareto

    "powerlaw",        #Power-function

    "powerlognorm",    #Power log normal

    "powernorm",       #Power normal

    "rdist",           #R distribution

    "reciprocal",      #Reciprocal

    "rayleigh",        #Rayleigh

    "rice",            #Rice

    "recipinvgauss",   #Reciprocal Inverse Gaussian

    "semicircular",    #Semicircular

    "t",               #Student's T

    "triang",          #Triangular

    "truncexpon",      #Truncated Exponential

    "truncnorm",       #Truncated Normal

    

#     "tukeylambda",     #Tukey-Lambda

    

    "uniform",         #Uniform

    "vonmises",        #Von-Mises (Circular)

    "wald",            #Wald

    "weibull_min",     #Minimum Weibull (see Frechet)

    "weibull_max",     #Maximum Weibull (see Frechet)

    "wrapcauchy",      #Wrapped Cauchy

    

#     "ksone",           #Kolmogorov-Smirnov one-sided (no stats)

#     "kstwobign"        #Kolmogorov-Smirnov two-sided test for Large N

    ]
for cdf in cdfs:

    # fit our data set against every probability distribution

    parameters = eval("stats."+cdf+".fit(data)")

    # applying the Kolmogorov-Smirnof test

    D, p = stats.kstest(data, cdf, args=parameters)

    # print

    print('p = %.25f, D = %.4f (%s)' % (p,D,cdf))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

import scipy.stats.distributions as scp

from scipy.optimize import curve_fit

from scipy.special import factorial

from scipy.stats import poisson

import sklearn.preprocessing as skl

import seaborn as sns

import warnings

import scipy.stats as st

import statsmodels as sm



Variables = ['video_id', 'views', 'likes', 'dislikes', 'comment_count']

        

us_yt = pd.read_csv('../input/youtube-new/USvideos.csv', usecols=Variables) #USA

ca_yt = pd.read_csv('../input/youtube-new/CAvideos.csv', usecols=Variables) #Canada

de_yt = pd.read_csv('../input/youtube-new/DEvideos.csv', usecols=Variables) #Germany

fr_yt = pd.read_csv('../input/youtube-new/FRvideos.csv', usecols=Variables) #France

gb_yt = pd.read_csv('../input/youtube-new/GBvideos.csv', usecols=Variables) #Great Brittain

in_yt = pd.read_csv('../input/youtube-new/INvideos.csv', usecols=Variables) #India

jp_yt = pd.read_csv('../input/youtube-new/JPvideos.csv', usecols=Variables) #Japan

kr_yt = pd.read_csv('../input/youtube-new/KRvideos.csv', usecols=Variables) #South Korea

mx_yt = pd.read_csv('../input/youtube-new/MXvideos.csv', usecols=Variables) #Mexico

ru_yt = pd.read_csv('../input/youtube-new/RUvideos.csv', usecols=Variables) #Russia



N = (us_yt.dtypes == 'int64')

Numeric = list(N[N].index)



us_yt['Country'] = "US"

ca_yt['Country'] = "CA"

de_yt['Country'] = "DE"

fr_yt['Country'] = "FR"

gb_yt['Country'] = "GB"

in_yt['Country'] = "IN"

jp_yt['Country'] = "JP"

kr_yt['Country'] = "KR"

mx_yt['Country'] = "MX"

ru_yt['Country'] = "RU"



#us_yt_sc = us_yt

#ca_yt_sc = ca_yt

#de_yt_sc = de_yt

#fr_yt_sc = fr_yt

#gb_yt_sc = gb_yt

#in_yt_sc = in_yt

#jp_yt_sc = jp_yt

#kr_yt_sc = kr_yt

#mx_yt_sc = mx_yt

#ru_yt_sc = ru_yt



# Center the data before computation of RV coefficients

#us_yt_sc[Numeric] = skl.scale(us_yt[Numeric], axis=0, with_mean=True)

#ca_yt_sc[Numeric] = skl.scale(ca_yt[Numeric], axis=0, with_mean=True)

#de_yt_sc[Numeric] = skl.scale(de_yt[Numeric], axis=0, with_mean=True)

#fr_yt_sc[Numeric] = skl.scale(fr_yt[Numeric], axis=0, with_mean=True)

#gb_yt_sc[Numeric] = skl.scale(gb_yt[Numeric], axis=0, with_mean=True)

#in_yt_sc[Numeric] = skl.scale(in_yt[Numeric], axis=0, with_mean=True)

#jp_yt_sc[Numeric] = skl.scale(jp_yt[Numeric], axis=0, with_mean=True)

#kr_yt_sc[Numeric] = skl.scale(kr_yt[Numeric], axis=0, with_mean=True)

#mx_yt_sc[Numeric] = skl.scale(mx_yt[Numeric], axis=0, with_mean=True)

#ru_yt_sc[Numeric] = skl.scale(ru_yt[Numeric], axis=0, with_mean=True)

 

df = pd.concat([us_yt, ca_yt, de_yt,fr_yt,gb_yt,in_yt,jp_yt,kr_yt,mx_yt,ru_yt] )

df.reset_index

df.head()
# Function for limiting extreme values

def ExtremeValues (Variable,DataSet,Quantile):

    DefinedCapped = DataSet[Variable].quantile(q=Quantile)

    DataSet.loc[DataSet[Variable] >= DefinedCapped, Variable] = DefinedCapped

    return DataSet
df_cap = ExtremeValues ('views',df,0.9)

df_cap = ExtremeValues ('likes',df,0.9)

df_cap = ExtremeValues ('dislikes',df,0.9)

df_cap = ExtremeValues ('comment_count',df,0.9)
n_bins = 100



fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')

ax0, ax1, ax2, ax3 = axes.flatten()



ax0.hist(df_cap['views'], n_bins, density=True, color = 'skyblue')

ax0.set_title("Views")



ax1.hist(df_cap['likes'], n_bins, density=True, color = 'yellow')

ax1.set_title("Likes")



ax2.hist(df_cap['dislikes'], n_bins, density=True, color = 'purple')

ax2.set_title("Dislikes")



ax3.hist(df_cap['comment_count'], n_bins, density=True, color = 'green')

ax3.set_title("comment_count")



plt.show()
mean1, var1  = scp.norm.fit(df_cap['views'])

x1 = np.linspace(0,2000000,1000000)

x1_fit = scp.norm.pdf(x1, mean1, var1)



plt.hist(df_cap['views'], density=True, color = 'skyblue')

plt.plot(x1,x1_fit,'r-')
df_views_lim = df.loc[df['views'] < 250000, 'views']

mean2, var2  = scp.norm.fit(df_views_lim)

x2 = np.linspace(0,250000,100000)

x2_fit = scp.norm.pdf(x2, mean2, var2)



plt.hist(df_views_lim, density=True, color = 'skyblue')

plt.plot(x2,x2_fit,'r-')
# Distributions to check 

AllDistributions = [ st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,

        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,

        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,

        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,

        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,

        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,

        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,

        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,

        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,

        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]



ChosenDistributions = [ st.alpha,st.beta,st.burr,st.cauchy,st.chi2,st.expon,st.gamma,st.gompertz,st.gumbel_r,st.laplace,

        st.levy,st.logistic,st.maxwell,st.norm,st.rayleigh,st.uniform]



ChosenDistributionsNames = [ st.alpha,st.beta,st.burr,st.cauchy,st.chi2,st.expon,st.gamma,st.gompertz,st.gumbel_r,st.laplace,

        st.levy,st.logistic,st.maxwell,st.norm,st.rayleigh,st.uniform]
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 15), dpi=80, facecolor='w', edgecolor='k')



for distribution, ax in zip(ChosenDistributions, axes.flat):

    params = distribution.fit(df_views_lim)

    arg = params[:-2]

    loc = params[-2]

    scale = params[-1]

    x = np.linspace(0,250000,len(df_views_lim))

    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)

    title_correct = (str(distribution).replace("<scipy.stats._continuous_distns.","")).split("_gen",1)[0]

    ax.hist(df_views_lim, bins=n_bins , density=True, color = 'skyblue')

    ax.plot(x,pdf,'r-')

    ax.set_title(title_correct)
for i in range(0,len(ChosenDistributionsNames)):

    ChosenDistributionsNames[i] = (str(ChosenDistributionsNames[i]).replace("<scipy.stats._continuous_distns.","")).split("_gen",1)[0]



Perf = pd.DataFrame([0.0000] * 16,columns=['SSE'],index=ChosenDistributionsNames)

Perf['Scale']=[0.0000] * 16

Perf['Location']=[0.0000] * 16

Perf['NoOtherParameters']=[0.0000] * 16
for distribution, i in zip(ChosenDistributions, range(0,len(ChosenDistributions))):   

    

    y, x = np.histogram(df_views_lim, bins=n_bins, density=True)

    x = (x + np.roll(x, -1))[:-1] / 2.0

    

    params = distribution.fit(df_views_lim)

    

    arg = params[:-2]

    loc = params[-2]

    scale = params[-1]

    

    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)

    Perf.SSE[i] = np.sum(np.power(y - pdf, 2.0)) * np.power(10, 10.0)

    Perf.Scale[i] = scale

    Perf.Location[i] = loc

    Perf.NoOtherParameters[i] = len(arg)
def color_negative_red(val):

    """

    Takes a scalar and returns a string with

    the css property `'color: red'` for negative

    strings, black otherwise.

    """

    color = 'red' if val < 0 else 'black'

    return 'color: %s' % color

def highlight_max(s):

    '''

    highlight the maximum in a Series yellow.

    '''

    is_max = s == s.max()

    return ['background-color: yellow' if v else '' for v in is_max]

def highlight_min(s):

    '''

    highlight the minimum in a Series yellow.

    '''

    is_min = s == s.min()

    return ['background-color: yellow' if v else '' for v in is_min]
Perf.sort_values(by="SSE").style.applymap(color_negative_red).apply(highlight_max).format({'SSE': "{:.6f}",'Scale': "{:.0f}",'Location': "{:.0f}",'NoOtherParameters': "{:.0f}"})
y, x = np.histogram(df_views_lim, bins=n_bins, density=True)

x = (x + np.roll(x, -1))[:-1] / 2.0



pdf_L = st.levy.pdf(x,Perf.Location[10],Perf.Scale[10])

pdf_E = st.expon.pdf(x, Perf.Location[5], Perf.Scale[5])



pdf_fin = (pdf_E + pdf_L)/2



Score = np.sum(np.power(y - pdf_fin, 2.0)) * np.power(10, 10.0)



print(format(Score))
pdf_L = st.levy.pdf(x,Perf.Location[10],Perf.Scale[10])

pdf_E = st.expon.pdf(x, Perf.Location[5], Perf.Scale[5])



MixWeights = pd.DataFrame([99.00]*11,columns=['SSE'], index=range(0,11))

MixList = [""]*11



for i in range(0,11):

    pdf_fin  =  pdf_E * i/10 + pdf_L * (1-i/10)

    MixWeights.SSE[i]= np.sum(np.power(y - pdf_fin, 2.0)) * np.power(10, 10.0)

    MixList[i]="Exponential: " + str(round(i/10,2)) + " Levy: " + str(round(1-i/10,2))



MixWeights=MixWeights.style.apply(highlight_min)

MixWeights
pdf_fin  =  pdf_E * 6/10 + pdf_L * (1-4/10)



fig, axes = plt.subplots(figsize=(10, 10))

axes.hist(df_views_lim, bins=n_bins , density=True, color = 'skyblue')

axes.plot(x,pdf_E,'g-',label='Exponential')

axes.plot(x,pdf_L,'y-',label='Levy')

axes.plot(x,pdf_fin,'r-',label='Mixture')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

axes.set_title('Mixture of exponential and Levy distributions')
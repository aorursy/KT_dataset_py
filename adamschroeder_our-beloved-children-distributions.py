import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import scipy as sp
import scipy.stats as st
import statsmodels as sm


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


import re
import warnings
warnings.filterwarnings('ignore')

child = pd.read_csv('../input/kiva_loans.csv')

child.disbursed_time = pd.to_datetime(child.disbursed_time)
child.posted_time = pd.to_datetime(child.posted_time)
child.funded_time = pd.to_datetime(child.funded_time)

child.head()
b = child['use'].apply(lambda x: len(re.findall(r"([\s.,:;]son[\s.,:;s])|([\s.,:;]son\s*[.'’]+)", str(x)))) 
g = child['use'].apply(lambda x: len(re.findall(r"([\s.,:;]daughter[\s.,:;s])|([\s.,:;]daughter\s*[.'’]+)", str(x)))) 

b = b[b.values>0]
g = g[g.values>0]

b = child.iloc[b.index,:]
g = child.iloc[g.index,:]
plt.figure(figsize=(10,6))
sns.distplot(b.funded_amount.values, bins=40, kde=False, color='black')
sns.distplot(g.funded_amount.values, bins=40, kde=False, color='yellow')
plt.title("Distribution of Loan (USD) for sons and daughters")
plt.legend(('sons','daughters'))
plt.ylabel("Number of loans")
plt.xlabel("Funded \$$")
plt.xlim((0,6000))
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
weights = np.ones_like(g.funded_amount.values)/len(g.funded_amount.values)
plt.hist(g.funded_amount.values, bins=10, weights=weights, color="yellow", alpha=0.9, normed=False) 
plt.ylabel("Proportion of Observation")
plt.xlabel('Funded \$$')
plt.title("PMF of funded loans")
plt.xlim((-100,6000))
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
sns.kdeplot(g.funded_amount.values, shade=True)
plt.xlim((-200,3500))
plt.title("Density of funded loans (PDF)")
plt.ylabel("Density")
plt.xlabel('Funded \$$')
plt.grid(True)
plt.show()
loan_posted = g.posted_time.apply(lambda x: x.hour)
plt.figure(figsize=(10,6))
sns.kdeplot(loan_posted.values, shade=True)
plt.title("Density of posted time of loan (PDF)")
plt.ylabel("Density")
plt.xlabel('Hour of Day')
plt.grid(True)
plt.show()
"""Compute ECDF for a one-dimensional array of measurements."""
def ecdf(data, datab):
    
    # Number of data points: n
    n = len(data)
    nb= len(datab)

    # x-data for the ECDF: x
    x = np.sort(data)
    xb= np.sort(datab)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    yb= np.arange(1, nb+1) / nb

    return x, y, xb, yb

"""Plot the ECDF"""
x, y, xb, yb = ecdf(g.funded_amount.values, b.funded_amount.values)

plt.figure(figsize=(10,6))
plt.plot(x,y, linestyle='none', marker='.', color='yellow', alpha=0.9)
plt.plot(xb,yb, linestyle='none', marker='.', color='gray', alpha=0.4)

plt.xlabel("Funded \$$")
plt.ylabel("Probability")
plt.title("CDF of funded loans")
plt.legend(('daughters','sons'))
plt.xlim((-200,6000))
plt.grid(True)
plt.show()
plt.figure(figsize=(10,6))
sns.kdeplot(g.funded_amount.values, shade=False, cumulative=True)
plt.xlim((-200,6000))
plt.xlabel("Funded \$$")
plt.ylabel("Probability")
plt.title("CDF of funded loans")
plt.grid(True)
plt.show()
# Big thanks to @tmthydvnprt from Stackoverflow for the code

# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.exponnorm,st.norm,st.cosine,st.dgamma,st.dweibull
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Propbability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

# Load data 
data = pd.Series(g.funded_amount.values)

# Plot for comparison
plt.figure(figsize=(12,8))
ax = data.plot(kind='hist', bins=50, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
# Save plot limits
dataYLim = ax.get_ylim()

# Find best fit distribution
best_fit_name, best_fir_paramms = best_fit_distribution(data, 200, ax)
best_dist = getattr(st, best_fit_name)

# Update plots
ax.set_ylim(dataYLim)
ax.set_title(u'All Fitted Distributions')
ax.set_xlabel(u'Funded \$$')
ax.set_ylabel('Frequency')

# Make PDF
pdf = make_pdf(best_dist, best_fir_paramms)

# Display
plt.figure(figsize=(12,8))
ax = pdf.plot(lw=2, label='PDF', legend=True)
data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)

param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fir_paramms)])
dist_str = '{}({})'.format(best_fit_name, param_str)

ax.set_title(u'Best fit distribution for daughters loans \n' + dist_str)
ax.set_xlabel(u'Funded \$$')
ax.set_ylabel('Frequency')
plt.show()
#Mean(‘m’), variance(‘v’), skew(‘s’), and/or kurtosis(‘k’)
print(st.exponnorm.stats(K=10.84, loc=154.35, scale=73.82, moments='mvsk'))
print("\nWe see that the mean of the fitted distribution ~ the mean of the empirical data:\n")
print('empirical mean:',g.funded_amount.values.mean())
st.exponnorm.stats(K=10.84, loc=154.35, scale=73.82, moments='mean')
# Generate 100,000 random samples of data within the Exponentially Modified Normal distribution, using the paramateres
# that represent the best fit.
theoretical_data_points = st.exponnorm.rvs(10.84, loc=154.35, scale=73.82, size=100000)

plt.figure(figsize=(10,6))
sns.kdeplot(g.funded_amount.values, shade=False, cumulative=True, color='black', label='Empirical data')
sns.kdeplot(theoretical_data_points, shade=False, cumulative=True, color='red', label='Theoretical data')
plt.xlim((-200,5000))
plt.xlabel("Funded \$$")
plt.ylabel("Probability")
plt.title("CDF of funded loans(theoretical/empirical)")
plt.legend()
plt.grid(True)
plt.show()
print("When fitting the data to the best distribution, we also get the distribution's shape, loc, scale")
st.exponnorm.fit(g.funded_amount.values)
theoretical_data_points = st.exponnorm.rvs(10.84, loc=154.35, scale=73.82, size=100000)

plt.figure(figsize=(10,6))
sns.kdeplot(theoretical_data_points, shade=True)
plt.xlim((-200,5000))
plt.title("Density of funded loans (PDF)")
plt.ylabel("Density")
plt.xlabel('Funded \$$')
plt.grid(True)
plt.show()
theory_std = st.exponnorm.std(K=10.84, loc=154.35, scale=73.82)
theory_mean = st.exponnorm.mean(K=10.84, loc=154.35, scale=73.82)

x = sp.linspace(-100,5*theory_std, 100000) # 100,000 numbers between -100 to 5 std to the right of mean. 
pdf = st.exponnorm.pdf(x, 10.84, loc=154.35, scale=73.82)
plt.figure(figsize=(10,6))
plt.plot(x,pdf, color="black")
plt.title("Density of funded loans (PDF)")
plt.xlabel("Funded \$$")
plt.ylabel("Density")
plt.xlim((-200,3000))
plt.grid(True)
plt.draw()
x = sp.linspace(-4*theory_std,4*theory_std, 100000)
cdf = st.exponnorm.cdf(x, K=10.84, loc=154.35, scale=73.82)
plt.figure(figsize=(7,4))
plt.plot(x,cdf, color="black")
plt.title("CDF funded loans")
plt.xlabel("Funded \$$")
plt.ylabel("Cumulative Probability")
plt.xlim((-200,3000))
plt.grid(True)
plt.show()
q = sp.linspace(0, 1.0, 100) # generate 100 numbers b/w 0.0-0.1
y = st.exponnorm.ppf(q, K=10.84, loc=154.35, scale=73.82)
plt.figure(figsize=(10,6))
plt.plot(q,y, color="black")
plt.title("PPF funded loans")
plt.xlabel("Cumulative Probability")
plt.ylabel("Funded \$$")
plt.grid(True)
plt.show()

print("For analysis of one percentile point, all we have to do is imput the percentile we care about in the code above.")
print("\nFor example, there is a 51% probability that a random data point (funded loan) will fall under: ${}".format(
st.exponnorm.ppf(0.51, K=10.84, loc=154.35, scale=73.82)))
x = sp.linspace(-3*theory_std,4*theory_std, 100000)
sf = st.exponnorm.sf(x, K=10.84, loc=154.35, scale=73.82)
plt.figure(figsize=(10,6))
plt.plot(x,sf, color="black")
plt.title("SF funded loans")
plt.xlabel("Funded \$$")
plt.ylabel("Cumulative Probability")
plt.xlim((-200,3000))
plt.grid(True)
plt.show()
print("The probability that a data point (funded loans) will fall above $728.58 is: {}".format(
round((st.exponnorm.sf(728.58, K=10.84, loc=154.35, scale=73.82)), 3)))
q = sp.linspace(0, 1.0, 100)
isf = st.exponnorm.isf(q, K=10.84, loc=154.35, scale=73.82)
plt.figure(figsize=(7,4))
plt.plot(q,isf, color="black")
plt.title("ISF funded loans")
plt.xlabel("Funded \$$")
plt.ylabel("Cumulative Probability")
plt.grid(True)
plt.show()
print("Here too, we can imput the percentile into code above.")
print("\nFor example, there is a 49% probability that a random data point (funded loan) will fall above: ${}".format(
st.exponnorm.isf(0.49, K=10.84, loc=154.35, scale=73.82)))
"""Annex A"""
Distributions = [st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy]
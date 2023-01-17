import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm
macro = pd.read_csv('../input/macro.csv')

train = pd.read_csv('../input/train.csv')
macro["timestamp"] = pd.to_datetime(macro["timestamp"])

macro["year"]  = macro["timestamp"].dt.year

macro["month"] = macro["timestamp"].dt.month

macro["yearmonth"] = 100*macro.year + macro.month

macmeds = macro.groupby("yearmonth").median()

macmeds.head()
train["timestamp"] = pd.to_datetime(train["timestamp"])

train["year"]  = train["timestamp"].dt.year

train["month"] = train["timestamp"].dt.month

train["yearmonth"] = 100*train.year + train.month

prices = train[["yearmonth","price_doc"]]

p = prices.groupby("yearmonth").median()

p.head()
df = macmeds.join(p)

# Take a look at some of the data, just to make sure it's there:

df.loc[ [201109,201212,201403,201506],

             ["cpi","balance_trade","mortgage_rate","year","month","price_doc"]]
#  Adapted from code at http://adorio-research.org/wordpress/?p=7595

#  Original post was dated May 31st, 2010

#    but was unreachable last time I tried



import numpy.matlib as ml

 

def almonZmatrix(X, maxlag, maxdeg):

    """

    Creates the Z matrix corresponding to vector X.

    """

    n = len(X)

    Z = ml.zeros((len(X)-maxlag, maxdeg+1))

    for t in range(maxlag,  n):

       #Solve for Z[t][0].

       Z[t-maxlag,0] = sum([X[t-lag] for lag in range(maxlag+1)])

       for j in range(1, maxdeg+1):

             s = 0.0

             for i in range(1, maxlag+1):       

                s += (i)**j * X[t-i]

             Z[t-maxlag,j] = s

    return Z



def almonXcof(zcof, maxlag):

    """

    Transforms the 'b' coefficients in Z to 'a' coefficients in X.

    """

    maxdeg  = len(zcof)-1

    xcof    = [zcof[0]] * (maxlag+1)

    for i in range(1, maxlag+1):

         s = 0.0

         k = i

         for j in range(1, maxdeg+1):

             s += (k * zcof[j])

             k *= i

         xcof[i] += s

    return xcof
y = df.price_doc.div(df.cpi).apply(np.log).loc[201108:201506]

print( y.head() )

y.shape
nobs = 47  # August 2011 through June 2015, months with price_doc data

tblags = 5    # Number of lags used on PDL for Trade Balance

mrlags = 5    # Number of lags used on PDL for Mortgage Rate

ztb = almonZmatrix(df.balance_trade.loc[201103:201506].as_matrix(), tblags, 1)

zmr = almonZmatrix(df.mortgage_rate.loc[201103:201506].as_matrix(), mrlags, 1)

columns = ['tb0', 'tb1', 'mr0', 'mr1']

z = pd.DataFrame( np.concatenate( (ztb, zmr), axis=1), y.index.values, columns )

X = sm.add_constant( z )

X.shape
eq = sm.OLS(y, X)

fit = eq.fit()

fit.summary()
%matplotlib inline

plt.plot(y.values)

plt.plot(pd.Series(fit.predict(X)).values)
test_cpi = df.cpi.loc[201507:201605]

test_index = test_cpi.index

ztb_test = almonZmatrix(df.balance_trade.loc[201502:201605].as_matrix(), tblags, 1)

zmr_test = almonZmatrix(df.mortgage_rate.loc[201502:201605].as_matrix(), mrlags, 1)

z_test = pd.DataFrame( np.concatenate( (ztb_test, zmr_test), axis=1), test_index, columns )

X_test = sm.add_constant( z_test )

pred_lnrp = fit.predict( X_test )

pred_p = np.exp(pred_lnrp) * test_cpi

pred_p.to_csv("monthly_macro_predicted.csv")

pred_p
print( "Here's the average price predicted for the test period by the macro model: \n")

print( np.exp( pred_lnrp.mean() + np.log(test_cpi).mean() ) )

print( "\nDivide (logarithmic) average baseline micro model price prediction by this")

print( "   and use the result to justify multiplier for training prices in the micro model.")
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from random import gauss

from random import seed

from pandas import Series

from pandas.tools.plotting import autocorrelation_plot

from matplotlib import pyplot

# seed random number generator

# create white noise series

train = pd.read_csv('../input/train.csv')



series = train['y']-train['y'].mean()

series = Series(series)

# summary stats

print(series.describe())

# line plot

series.plot()

pyplot.show()

# histogram plot

series.hist()

pyplot.show()

# autocorrelation

autocorrelation_plot(series)

pyplot.show()
#Stock market behaves identical..

#-----

train = pd.read_csv('../input/train.csv')

y = np.array(train['y'])

test = pd.read_csv('../input/test.csv')



# Analyze all features ; modify categorical features



train['y_mean'] = train.copy().groupby('X0')['y'].transform('mean')

train['y_max'] = train.copy().groupby('X0')['y'].transform('max')

train['y_min'] = train.copy().groupby('X0')['y'].transform('min')

train['y_median'] = train.copy().groupby('X0')['y'].transform('median')

#print(train)

train_test = train.append(test)

train_test= train_test.sort_values(by='ID')



train_test=train_test[['y','ID','X0','X314','X5','X8']]

labels_X0=set( train['X0'] )

#print(train_test)

n_assets=len(labels_X0)

def rand_weights(n):

    k = np.random.rand(n)

    return k / sum(k)



train_test['tijdzone']=round(train_test['ID']/48,0)

train_pos=train_test[train_test['y']>0]

#print(train_pos)

tr_te_pima = pd.pivot_table(train_pos, values='y', index=['X0'],columns=['tijdzone'], aggfunc=np.max)

tr_te_pimi = pd.pivot_table(train_pos, values='y', index=['X0'],columns=['tijdzone'], aggfunc=np.min)



ri_re_pl=pd.DataFrame([])



ri_re_pl['re']=tr_te_pima.sum()

ri_re_pl['ri']=tr_te_pima.std()

ri_re_pl['nr']=tr_te_pima.count()

ri_re_pl['ti']=ri_re_pl['re']/ri_re_pl['nr']

ri_re_pli=pd.DataFrame([])

ri_re_pli['re']=tr_te_pimi.sum()

ri_re_pli['ri']=tr_te_pimi.std()

ri_re_pli['nr']=tr_te_pima.count()

ri_re_pli['ti']=ri_re_pli['re']/ri_re_pli['nr']



#print(ri_re_pl)

print('Plotting time spend max (blue) - min (orange) time per 48 cars versus variability')

import numpy as np

import matplotlib.pyplot as plt

plt.scatter(ri_re_pl['ri'],ri_re_pl['ti'], alpha=0.5)

plt.scatter(ri_re_pli['ri'],ri_re_pli['ti'], alpha=0.5)

plt.show()
import cvxopt as opt

from cvxopt import blas, solvers





def optimal_portfolio(returns):

    n = len(returns)

    returns = np.asmatrix(returns)

    

    N = 100

    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]

    

    # Convert to cvxopt matrices

    S = opt.matrix(np.cov(returns))

    pbar = opt.matrix(np.mean(returns, axis=1))

    

    # Create constraint matrices

    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix

    h = opt.matrix(0.0, (n ,1))

    A = opt.matrix(1.0, (1, n))

    b = opt.matrix(1.0)

    

    # Calculate efficient frontier weights using quadratic programming

    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 

                  for mu in mus]

    ## CALCULATE RISKS AND RETURNS FOR FRONTIER

    returns = [blas.dot(pbar, x) for x in portfolios]

    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]

    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE

    m1 = np.polyfit(returns, risks, 2)

    x1 = np.sqrt(m1[2] / m1[0])

    # CALCULATE THE OPTIMAL PORTFOLIO

    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return np.asarray(wt), returns, risks



weights, returns, risks = optimal_portfolio(ri_re_pl)



plt.plot(stds, means, 'o')

plt.ylabel('mean')

plt.xlabel('std')

plt.plot(risks, returns, 'y-o')
# fast group

print(ri_re_pli[ri_re_pl['ti']<98])

#slow group

print(ri_re_pli[ri_re_pl['ti']>109])
import seaborn as sns

#list of cars with short time

fast=train_test[train_test['tijdzone']==82]

fast=fast.groupby(['X0']).count()

sns.set_style("whitegrid")

ax = sns.barplot(x=fast.index, y="ID", data=fast)

fast=train_test[train_test['tijdzone']==144]

fast=fast.groupby(['X0']).count()

sns.set_style("whitegrid")

ax = sns.barplot(x=fast.index, y="ID", data=fast)



fast=train_test[train_test['tijdzone']==61]

fast=fast.groupby(['X0']).count()

sns.set_style("whitegrid")

ax = sns.barplot(x=fast.index, y="ID", data=fast)

#list of cars with long time

fast2=train_test[train_test['tijdzone']==68]

fast2=fast2.groupby(['X0']).count()

aax = sns.barplot(x=fast2.index, y="ID", data=fast2)

#list of cars with long time

fast2=train_test[train_test['tijdzone']==104]

fast2=fast2.groupby(['X0']).count()

aax = sns.barplot(x=fast2.index, y="ID", data=fast2)
#list of cars with long time

fast2=train_test[train_test['tijdzone']==114]

fast2=fast2.groupby(['X0']).count()

aax = sns.barplot(x=fast2.index, y="ID", data=fast2)
#print(train_test)

pivot=pd.pivot_table(train_test[train_test['y']>0], values='y', index=['X0'],columns=['tijdzone'], aggfunc=np.median)

# not 1OO% happy with that ffill solution, but for the time being that proofs the model

print(pivot.fillna( method='ffill', axis=1, inplace=True) )

print(pivot.fillna( method='bfill', axis=1, inplace=True) )

# here is the variance covariance matrix, but some are '0.0' should be dropped

vcm=pivot.T.cov()

vcm=vcm.drop(['ab','ac','g'])

vcm=vcm.drop(['ab','ac','g'],axis=1)

vcm= vcm.append(pd.DataFrame ( [1 for x in range (0,len(vcm))],index=vcm.index,columns=['mark'] ).T ) 

vcm['mark']=-1.0

print(vcm)

# set the corner to a target value...





vcm.ix['mark','mark']=100.0

print(vcm.ix['mark','mark'])



# invert

df_sol = pd.DataFrame(np.linalg.pinv(vcm.values), vcm.columns, vcm.index)

# look at last column

print(df_sol['mark']*480)



#ok i forgot this one, the optimisation with the matrixinversion gives negatives values, that are in portfolio theorie shorting positions, but in real life i wonder how Benz could short a car?



#give me some time to solve this one, i have to read abit
def rand_weight(m,n):

    s = np.random.normal(5,2, int(n/2))

    s = s.round()

    r = np.random.rand(m)*m

    r = r.round()

    d = [t ]

    return r,s    



print(rand_weight(28,20))

print(rand_weight(28,20))
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, kurtosistest, norm, t

import matplotlib.pyplot as plt
from matplotlib import cm
%matplotlib inline
import bokeh
from bokeh.io import output_notebook
from bokeh.layouts import column
from bokeh.palettes import inferno
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, Legend, Span, Label
output_notebook()

import tqdm
data = pd.read_csv("../input/crypto-markets.csv")
close_df = pd.pivot_table(data, index='date', columns='symbol', values='close',
                              fill_value=0.0).loc['2015-01-01':]

price = close_df.BTC.astype('f').dropna()
ret = price.pct_change().dropna()

plt.figure(figsize=(12,4))
plt.title("Preço da Bitcoin em USD")
plt.ylabel("preço")
plt.plot(pd.to_datetime(price.index), price)

plt.figure(figsize=(12,4))
plt.title("Retornos da Bitcoin em %")
plt.ylabel("retornos")
plt.plot(pd.to_datetime(ret.index), ret * 100);
grey = .66, .66, .77
plt.figure(figsize=(13,6))
plt.title("Distribuição de retornos")
plt.ylabel("frequência")
plt.xlabel("magnitude")
plt.hist(ret, bins=50, normed=True, color=grey, edgecolor='none');
alpha1 = 0.05

sorted_ret = np.sort(ret.ravel())

var95 = abs(sorted_ret[int((1 - alpha1) * sorted_ret.shape[0])])
print("%.d%% Empirical VaR: %.2f %%" % ((
    1 - alpha1) * 100,
    var95 * 100)
 )

grey = .66, .66, .77
plt.figure(figsize=(13,6))
plt.title("Distribuição de retornos")
plt.ylabel("frequência")
plt.xlabel("magnitude")
plt.hist(ret, bins=50, normed=True, color=grey, edgecolor='none');

plt.plot([-var95, -var95], [0, 6], c='r')
plt.text(-var95-0.01, 7.5, "95%% VaR", color='r')
plt.text(-var95-0.01, 6.5, -var95, color='r')

plt.grid(linestyle='--')
# VaR
alpha1 = 0.05

sorted_ret = np.sort(ret.values.ravel())
index = int(alpha1 * sorted_ret.shape[0])
var95 = abs(sorted_ret[index])
print("%.d%% Empirical VaR: %.2f %%" % ((
    1 - alpha1) * 100,
    var95 * 100)
 )

# CVaR
# Calculate the total VaR beyond alpha
sum_var = sorted_ret[:index].sum()
# Return the average VaR
# CVaR should be positive
cvar95 = abs(sum_var / index)

print("%.d%% Empirical CVaR: %.2f %%" % ((
    1 - alpha1) * 100,
    cvar95 * 100)
 )

# plot results
grey = .66, .66, .77
plt.figure(figsize=(13,6))
plt.title("Distribuição de retornos")
plt.ylabel("frequência")
plt.xlabel("magnitude")
N, bins, patches = plt.hist(ret, bins=100, normed=True, color=grey, edgecolor='none');

for i in range(1, len(bins)):
    if bins[i] > -var95:
        break
    patches[i-1].set_facecolor('red')
    
plt.plot([-var95, -var95], [0, 6], c='r')
plt.text(-var95-0.01, 7.5, "95% VaR", color='r')
plt.text(-var95-0.01, 6.5, "%.6f" % -var95, color='r')

plt.plot([-cvar95, -cvar95], [0, 6], c='r')
plt.text(-cvar95-0.03, 7.5, "95% CVaR", color='r')
plt.text(-cvar95-0.03, 6.5, "%.6f" % -cvar95, color='r')

plt.grid(linestyle='--')
start_date = '2016-01-01'
reevaluate_freq = 7
n = 20

data = pd.read_csv("../input/crypto-markets.csv")
marketcapDf = pd.pivot_table(data, index='date', columns='symbol', values='market',
                              fill_value=0.0).loc[start_date:]
marketcapDf = marketcapDf.apply(lambda x: [x[i] / x.sum() for i in range(x.shape[0])],
                                  axis=1).astype('f')
closeDf = pd.pivot_table(data, index='date', columns='symbol', values='close').astype('f').fillna(0).loc[start_date:]
smoothed_mc = marketcapDf.ewm(alpha=0.05).mean()
# Select largest marketcap coins
largest = pd.DataFrame(index=smoothed_mc.columns)
for i, row in enumerate(smoothed_mc.iterrows()):
    # choose pairs every quarter
    if i % int(reevaluate_freq) == 0:
        components_index = np.argpartition(row[1].values, -n)[-n:]
    
    largest[row[0]] = row[1].iloc[components_index].fillna(0)
largest = largest.T
largest.shape
for col in largest.columns:
    if largest[col].isnull().all():
        largest = largest.drop(col, axis=1)
        closeDf = closeDf.drop(col, axis=1)
retDf = closeDf.fillna(0).rolling(2).apply(lambda x: np.clip(x[-1] / x[-2], 0, 2)).fillna(1).replace([np.inf, -np.inf], 1)
largest.shape, closeDf.shape, retDf.shape
largest.iloc[-1].nlargest(20).index
alpha1 = 0.05

cvar = {}
for symbol in largest.iloc[-1].nlargest(20).index:

    sorted_ret = np.sort(retDf[symbol] - 1)
    index = int(alpha1 * sorted_ret.shape[0])
    var95 = abs(sorted_ret[index])
#     print("%.d%% Empirical VaR for %s: %.2f %%" % ((
#         1 - alpha1) * 100,
#         symbol,
#         var95 * 100)
#      )

    # CVaR
    # Calculate the total VaR beyond alpha
    sum_var = sorted_ret[:index].sum()
    # Return the average VaR
    # CVaR should be positive
    cvar95 = abs(sum_var / index)
    cvar[symbol] = cvar95
    print("%.d%% Empirical CVaR for %s: %.2f %%" % ((
        1 - alpha1) * 100,
        symbol,
        cvar95 * 100)
     )
def rebalance(oldWeights, relativeStrength, mpc=.1):
    # Optimization constraints
    constraints = [
        # analitycal value
#         {'type': 'eq', 'fun': lambda w: np.dot(oldWeights, close) - np.dot(w, close)},
        # Simplex constraints
        {'type': 'eq', 'fun': lambda w: w.sum() - 1},
    ]

    bounds = []
    for item in relativeStrength:
        if not np.allclose(0.0, item, rtol=1e-8, atol=1e-8):
            bounds.append((0, mpc))
        else:
            bounds.append((0,0))
    
    b = minimize(
        lambda w: np.linalg.norm(relativeStrength - w),
        relativeStrength,
        tol=None,
        constraints=constraints,
        bounds=bounds,
        jac=False,
        method='SLSQP',
        options = {'disp': False,
                   'iprint': 1,
                   'eps': 1.4901161193847656e-08,
                   'maxiter': 666,
                   'ftol': 1e-12
                   }
    )
    
    return b['x'], b['fun'], b['nit']
# Daily rebalancing
sqrtDom = largest.fillna(0).apply(lambda x: np.sqrt(x) / np.sqrt(x).sum(), axis=1).replace([np.inf, -np.inf], 1)

plt.figure(figsize=(13,9))
for col in sqrtDom.columns:
    if col in list(largest.iloc[-1].dropna().index):
        plt.plot(pd.to_datetime(sqrtDom.index), sqrtDom[col].values, label=col)
    else:
        plt.plot(pd.to_datetime(sqrtDom.index), sqrtDom[col].values)
plt.legend()
plt.title("Raiz da força relativa")
plt.xlabel("tempo")
plt.ylabel("% mercado");
cmi20 = []
weights = [sqrtDom.iloc[0].values]
# rebalance portfolio every day
for i, row in enumerate(sqrtDom.iterrows()):
    ret = retDf.iloc[i].values
    relativeStrength = row[1].values
    # recalculate weights every quarter
    if i % int(reevaluate_freq) == 0:
        w, loss, niter = rebalance(weights[-1], relativeStrength)
        print(i, loss, niter)
        weights.append(w)
        
    cmi20.append((largest.index[i], np.dot(weights[-1], ret)))
    
cmi20 = pd.DataFrame.from_records(cmi20).set_index(0, drop=True).cumprod()
# Index performance
plt.figure(figsize=(12,8))
# plt.plot(largest.index, np.log10(index_value));

plt.plot(pd.to_datetime(largest.index), cmi20, label='CMI20')

# plt.plot(pd.to_datetime(largest.index), closeDf.BTC / closeDf.BTC.iloc[0], label='BTC')

plt.title("Desempenho CMI20")
plt.ylabel("desempenho")
plt.xlabel("tempo")
plt.yscale('log')
plt.legend()
plt.grid(True, which='both');
ret = cmi20.pct_change().fillna(0).values.ravel()
sorted_ret = np.sort(ret)
index = int(alpha1 * sorted_ret.shape[0])
var95 = abs(sorted_ret[index])

# CVaR
# Calculate the total VaR beyond alpha
sum_var = sorted_ret[:index].sum()
# Return the average VaR
# CVaR should be positive
cvar95 = abs(sum_var / index)

print("%.d%% Empirical CVaR for %s: %.2f %%" % ((
    1 - alpha1) * 100,
    "CMI20",
    cvar95 * 100)
 )

cvar["CMI20"] = cvar95

# plot results
grey = .66, .66, .77
plt.figure(figsize=(13,6))
plt.title("Distribuição de retornos")
plt.ylabel("frequência")
plt.xlabel("magnitude")
N, bins, patches = plt.hist(ret, bins=100, normed=True, color=grey, edgecolor='none');

for i in range(len(bins)):
    if bins[i] > -var95:
        break
    patches[i-1].set_facecolor('red')
    
plt.plot([-cvar95, -cvar95], [0, 6], c='r')
plt.text(-cvar95-0.03, 7.5, "95% CVaR", color='r')
plt.text(-cvar95-0.03, 6.5, "%.6f" % -cvar95, color='r')

plt.grid(linestyle='--')
import operator
sorted_cvar = sorted(cvar.items(), key=operator.itemgetter(1))

keys = []
values = []
for item in sorted_cvar:
    keys.append(item[0])
    values.append(item[1])

plt.figure(figsize=(12,6))
plt.grid(linestyle='--')
bars = plt.bar(keys, values);
bars[2].set_facecolor('green')
plt.plot(keys, [cvar['CMI20']] * len(keys), color='r');

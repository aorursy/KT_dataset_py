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
marketcap_df = pd.pivot_table(data, index='date', columns='symbol', values='market',
                              fill_value=0.0).loc['2017-01-01':]

plt.figure(figsize=(13,6))
marketcap_df.sum(axis=1).divide(1000000000).plot()
plt.title("Capitalização do mercado")
plt.ylabel("valor ($ Bi)")
plt.xlabel("data")
plt.grid(linestyle='--', alpha=0.5);
marketcap_df = marketcap_df.apply(lambda x: [x[i] / x.sum() for i in range(x.shape[0])],
                                  axis=1).astype('f')

smoothed_mc = marketcap_df.ewm(alpha=0.1).mean()

n = 20
largest = pd.DataFrame(index=marketcap_df.columns)
for i, row in enumerate(smoothed_mc.iterrows()):
    # choose pairs every quarter
    if i % int(7) == 0:
        components_index = np.argpartition(row[1].values, -n)[-n:]
    largest[row[0]] = row[1].iloc[components_index].fillna(0)
largest = largest.T
pie = largest.iloc[-1].dropna().sort_values()
pie['Outras moedas'] = 1 - largest.iloc[-1].sum()
fig1, ax1 = plt.subplots(figsize=(8,8))
ax1.pie(pie.values, labels = pie.index, radius=1.5, autopct='%.1f %%', pctdistance=0.8, shadow=True, explode=np.append(np.zeros(len(pie)-1),0.2))
ax1.axis('equal');
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
plt.plot(pd.to_datetime(ret.index), ret);
grey = .66, .66, .77
plt.figure(figsize=(13,6))
plt.title("Distribuição de retornos")
plt.ylabel("frequência")
plt.xlabel("magnitude")
plt.hist(ret, bins=50, normed=True, color=grey, edgecolor='none');
alpha1 = 0.05
alpha2 = 0.01

sorted_ret = np.sort(ret)

mu = ret.mean()
print("Média da distribuição: %.4f %%" % 
        mu
 )

var95 = abs(sorted_ret[int((1 - alpha1) * sorted_ret.shape[0])])
print("%.d%% Empirical VaR: %.2f %%" % ((
    1 - alpha1) * 100,
    var95 * 100)
 )

var99 = abs(sorted_ret[int((1 - alpha2) * sorted_ret.shape[0])])
print("%.d%% Empirical VaR: %.2f %%" % ((
    1 - alpha2) * 100,
    var99 * 100)
 )

grey = .66, .66, .77
plt.figure(figsize=(13,6))
plt.title("Distribuição de retornos")
plt.ylabel("frequência")
plt.xlabel("magnitude")
plt.hist(ret, bins=50, normed=True, color=grey, edgecolor='none');
plt.text(mu+0.003, 23, "Média: %.4f" % mu, color='black')
plt.plot([mu, mu], [0, 24], c='black')
plt.plot([-var95, -var95], [0, 6], c='b')
plt.text(-var95-0.01, 7.5, "95%% VaR", color='b')
plt.text(-var95-0.01, 6.5, var95, color='b')
plt.plot([-var99, -var99], [0, 4], c='r')
plt.text(-var99-0.01, 5.5, "99%% VaR", color='r')
plt.text(-var99-0.01, 4.5, var99, color='r');
plt.grid(linestyle='--')
dx = 0.0001  # resolution
x = np.arange(-5, 5, dx)
pdf = norm.pdf(x, 0, 1)

plt.figure(figsize=(13, 5))
plt.plot(x, pdf, 'b')
plt.title("Distribuição normal")
plt.ylabel("frequência")
plt.xlabel("magnitude")
plt.text(0.05, 0.42, "Média", color='black')
plt.plot([0, 0], [0, 0.43], c='black')
cred = norm.ppf(1-0.01) * 1 - 0
plt.plot([-cred, -cred], [0, 0.11], c='r')
plt.text(-cred-0.9, 0.1, "95%% cred", color='r')
plt.grid(linestyle='--')
mu_norm, sig_norm = norm.fit(ret)
dx = 0.0001  # resolution
x = np.arange(-0.2, 0.2, dx)
pdf = norm.pdf(x, mu_norm, sig_norm)

var95 = norm.ppf(1 - alpha1) * sig_norm - mu_norm
var99 = norm.ppf(1 - alpha2) * sig_norm - mu_norm

print("%.d%% Normal VaR: %.2f %%" % ((
    1 - alpha1) * 100,
    var95 * 100)
     )
print("%.d%% Normal VaR: %.2f %%" % ((
    1 - alpha2) * 100,
    var99 * 100)
     )

plt.figure(figsize=(13, 5))
plt.plot(x, pdf, 'b')
plt.title("Distribuição normal")
plt.ylabel("frequência")
plt.xlabel("magnitude")
plt.text(mu_norm+0.003, 10.5, "Média: %.4f" % mu_norm, color='black')
plt.plot([mu_norm, mu_norm], [0, 11], c='black')
plt.plot([-var95, -var95], [0, 4.2], c='r')
plt.text(-var95-0.035, 4.5, "95%% VaR", color='r');
plt.text(-var95-0.025, 4, "%.4f" % var95, color='r');
plt.plot([-var99, -var99], [0, 2.2], c='r')
plt.text(-var99-0.035, 2.5, "99%% VaR", color='r');
plt.text(-var99-0.025, 2, "%.4f" % var99, color='r');
plt.grid(linestyle='--')
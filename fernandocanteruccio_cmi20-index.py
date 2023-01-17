import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
start_date = '2015-01-01'
reevaluate_freq = 7
n = 20

data = pd.read_csv("../input/crypto-markets.csv")
marketcapDf = pd.pivot_table(data, index='date', columns='symbol', values='market',
                              fill_value=0.0).loc[start_date:]
marketcapDf = marketcapDf.apply(lambda x: [x[i] / x.sum() for i in range(x.shape[0])],
                                  axis=1).astype('f')
closeDf = pd.pivot_table(data, index='date', columns='symbol', values='close').astype('f').fillna(0).loc[start_date:]
words = {}
for symbol in data.symbol.unique():
    words[symbol] = data.query("symbol == '%s'" % symbol).market.iloc[-1]
from wordcloud import WordCloud
 
wc = WordCloud(width=1024, height=500).generate_from_frequencies(words, 400);
 
plt.figure(figsize=(12,12));
plt.imshow(wc, interpolation='bilinear');
plt.tight_layout(pad=4);
plt.axis("off");
plt.figure(figsize=(13,13))
plt.pie([float(value) for value in words.values()], labels=list(words.keys()),  autopct='%1.1f%%');
plot = False

smoothed_mc = marketcapDf.ewm(alpha=0.05).mean()

if plot:
    mc_fig = figure(title="Preponderância media",
                   x_axis_type="datetime",
                   x_axis_label='date',
                   y_axis_label='Capital',
                   plot_width=900, plot_height=500,
                   tools=['crosshair','reset','xwheel_zoom','pan,box_zoom', 'save'],
                   toolbar_location="above"
               );

    palette = inferno(256)
    index = pd.to_datetime(smoothed_mc.index)
    for i, s in tqdm.tqdm((enumerate(smoothed_mc.columns))):
        mc_fig.line(
            index,
            smoothed_mc.iloc[:, i].values,
            color=palette[i % 255]
        );

    show(mc_fig);
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
# Relative strength
plt.figure(figsize=(13,9))
for col in largest.columns:
    if col in list(largest.iloc[-1].dropna().index):
        plt.plot(pd.to_datetime(largest.index), largest[col].values, label=col)
    else:
        plt.plot(pd.to_datetime(largest.index), largest[col].values)
plt.legend()
plt.title("Força relativa")
plt.xlabel("tempo")
plt.ylabel("% mercado");
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
# Index weights over time
plt.figure(figsize=(13,6))
plt.subplot(121)
plt.plot(weights[1:])
plt.title("Distribuição histrica")
plt.ylabel("% portfolio")
plt.xlabel("rebalanços")
plt.grid()

cols = sqrtDom.iloc[-1].notnull()
symbols = sqrtDom.iloc[-1][cols]
symbols = list(symbols.index)
w = pd.Series(index=sqrtDom.columns, data=weights[-1])[symbols].sort_values()

plt.subplot(122)
plt.pie(w, labels=w.index,  autopct='%1.1f%%')
plt.title("Distribuição atual");
# marketcap captured by the cmi20 index
plt.figure(figsize=(13,7))
largest.fillna(0).sum(axis=1).plot()
plt.title("Parcela do capital descrita")
plt.xlabel("tempo")
plt.ylabel("% mercado");
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
weekly_return = (cmi20.apply(lambda x: x[-1] / x[-8]).values[0] - 1) * 100
monthly_return = (cmi20.apply(lambda x: x[-1] / x[-29]).values[0] - 1) * 100
quaterly_return = (cmi20.apply(lambda x: x[-1] / x[-92]).values[0] - 1) * 100
halfyearly_return = (cmi20.apply(lambda x: x[-1] / x[-(92 * 2)]).values[0] - 1) * 100
yearly_return = (cmi20.apply(lambda x: x[-1] / x[-366]).values[0] - 1) * 100

msg =  'Retorno total:\n\n'
# msg += 'Semana:       %.2f %%\n' % weekly_return
msg += 'Mês:          %6.2f %%\n' % monthly_return
msg += 'Trimestre:    %6.2f %%\n' % quaterly_return
msg += 'Semestre:     %6.2f %%\n' % halfyearly_return
msg += 'Ano:          %6.2f %%\n' % yearly_return

print(msg)

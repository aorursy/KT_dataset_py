!pip install trendln yfinance
import trendln

import yfinance as yf

import numpy as np

import matplotlib.pyplot as plt

import altair as alt

import pandas as pd



tick = yf.Ticker("TSLA")

hist = tick.history(period="max", rounding=True)

hist = hist["2010-01-01":]
price = hist["Close"]

low52 = min(price[-52 * 5 :])

high52 = max(price[-52 * 5 :])



plot_price = alt.Chart(hist.reset_index()).mark_line().encode(x="Date", y="Close")



overlay = pd.DataFrame({"Close": [low52]})

plot_low52 = (

    alt.Chart(overlay).mark_rule(color="gray", strokeWidth=2).encode(y="Close:Q")

)



overlay = pd.DataFrame({"Close": [high52]})

plot_high52 = (

    alt.Chart(overlay).mark_rule(color="gray", strokeWidth=2).encode(y="Close:Q")

)
alt.layer(plot_price, plot_low52 + plot_high52).interactive()
hist = hist[-200*5:]



from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 6), dpi=100, facecolor='w', edgecolor='w')



fig = trendln.plot_sup_res_date(hist.Close, hist.index, window=125, fromwindows = False)

plt.savefig('support_resistance.svg', format='svg')

plt.show()

plt.clf();
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style("white")

(pd.read_csv("../input/corn2013-2017.txt", header=None)

     .rename(columns={0: 'Date', 1: 'Price ($/bushel)'})

     .set_index('Date')

     .plot.line(figsize=(14, 6), fontsize=14)

)



plt.title("Corn Price over Time", fontsize=20)



sns.despine()
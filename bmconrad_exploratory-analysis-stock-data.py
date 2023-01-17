import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output



rootPath = "../input/"

file_names = [rootPath + i for i in check_output(["ls", "../input"]).decode("utf8").split("\n")[:-1]]

print(file_names)



data_frames = [pd.read_csv(i) for i in file_names]

mainFrame = data_frames[1]



# Set the stage!

mainFrame["change"] = mainFrame["close"] - mainFrame["open"]

df_interest = mainFrame[["symbol", "date", "open", "close", "change", "volume"]]

df_interest["date"] = pd.to_datetime(df_interest["date"])

df_interest.head()
# Wide long -- PIVOT

symbolCloseWide = mainFrame.pivot(index='symbol', 

                                  columns='date', 

                                  values='close')

symbolCloseWide.head()

# Long wide -- MELT

symbolCloseWide["symbol"] = symbolCloseWide.index

newdf = pd.DataFrame(symbolCloseWide.as_matrix(),

                   columns=list(symbolCloseWide.columns))

symbolCloseLong = pd.melt(newdf, 

              id_vars=["symbol"], var_name="date", value_name="close",

              value_vars=list(newdf.columns[:-1]))

symbolCloseLong.head()
# Look at some stocks over time

symbols = df_interest["symbol"].unique().tolist()

for u in symbols[:10]:

    dates = df_interest[(df_interest["symbol"] == u)]["date"]

    values = df_interest[(df_interest["symbol"] == u)]["close"]

    plt.plot(dates.tolist(), values.tolist())

plt.legend(symbols, loc='upper left')

plt.show()

latest_year = max(pd.unique(list(df_interest["date"].apply(lambda x:x.year))))

latest_year_mask = [i==latest_year for i in df_interest["date"].apply(lambda x:x.year)]

df_masked1 = df_interest[latest_year_mask]

df_masked1.head()
symbols = df_interest["symbol"].unique().tolist()

maxPerSymbol={}

for u in symbols:

    values = df_interest[(df_interest["symbol"] == u)]["close"].tolist()

    maxPerSymbol[u] = max(values)



top10Closers = list(dict(sorted(maxPerSymbol.items(), 

                   key=lambda v:v[1], #sort by key -> v[0] | sort by value->v[1]

                   reverse=True)[:10]).keys())



top_10_mask = [i in top10Closers for i in df_masked1["symbol"].tolist()]

df_masked2 = df_masked1[top_10_mask]

df_masked2.head()
symbols = df_masked2["symbol"].unique().tolist()

for u in symbols[:10]:

    dates = df_masked2[(df_interest["symbol"] == u)]["date"]

    values = df_masked2[(df_interest["symbol"] == u)]["close"]

    plt.plot(dates.tolist(), values.tolist())

#plt.legend(symbols, loc='upper left')

plt.legend(symbols, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title(r'Top 10 Closing Stocks in 2016', fontsize=16)



plt.xlabel('Time (years)', fontsize=12)

plt.ylabel('Price ($/stock)', fontsize=12)

plt.show()
def build_density_facetwrap(somedf, colName, valName, wrapAmount, yourTitle):

    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})



    # Initialize the FacetGrid object

    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)

    g = sns.FacetGrid(somedf,col=colName, hue=colName, col_wrap=wrapAmount, palette=pal)

    # Draw the densities in a few steps

    g.map(sns.kdeplot, valName, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)

    g.map(sns.kdeplot, valName, clip_on=False, color="w", lw=0.5, bw=.2)

    g.map(plt.axhline, y=0, lw=2, clip_on=False)





    # Define and use a simple function to label the plot in axes coordinates

    def label(x, color, label):

        ax = plt.gca()

        ax.text(0, .4, label, fontweight="bold", color=color, 

                ha="left", va="center", transform=ax.transAxes)

        plt.xlabel('Change ($/day)', fontsize=12)

        plt.ylabel('Frequency', fontsize=12)



    g.map(label, "change")



    # Set the subplots to overlap

    g.fig.subplots_adjust(hspace=0)



    # Remove axes details that don't play will with overlap





    g.set_titles("")

    g.fig.suptitle(yourTitle, fontsize=32)

    g.set(yticks=[])

    g.despine(bottom=True, left=True)

    g.fig.subplots_adjust(top=.8)

    g.add_legend()



build_density_facetwrap(df_masked2,

                       "symbol",

                        "change",

                        5,

                        "Top 10 Closers Change Spread in 2016"

                       )

df_masked2["date"] = pd.to_datetime(df_masked2["date"])

months = list(df_masked2["date"].apply(lambda x:x.month).unique())

df_masked2["month"] = df_masked2["date"].apply(lambda x: x.month)

from __future__ import division



# Get each symbols standard deviation in time

ss = df_interest.groupby(by=["symbol"])["change"].std()



# Count each symbol groups positive day change in time

## NOTE: it is a proportion to account for some days not appearing for some symbols in time

pcs = df_interest.groupby(by='symbol').apply(lambda grp: grp[grp['change'] > 0]['change'].count() / grp['change'].size)



avgv = df_interest.groupby(by=['symbol'])['volume'].mean()/10000000



newdf = pd.concat([ss, pcs, avgv], axis=1).reset_index()

newdf.columns = ['symbol', 'std', 'prop_pos_day_change', "avg_volume"]

newdf.head()
for i in newdf['symbol'].tolist():

    x = newdf[newdf['symbol'] == i]['std']

    y = newdf[newdf['symbol'] == i]['prop_pos_day_change']

    plt.scatter(x,y)

plt.legend(newdf['symbol'].tolist(),

           bbox_to_anchor=(1.05, 1),

           loc=2,

           borderaxespad=0.,

          ncol=10)





plt.title(r'Stock Change and Spread', fontsize=32)

plt.xlabel('Daily Positive Change (%)', fontsize=22)

plt.ylabel('Total Standard Devation', fontsize=22)

plt.figure(figsize=(10,100))

plt.show()
from sklearn.cluster import KMeans



# Visualize K = {3..9}

kValues = [i for i in range(3,10)]

for k in kValues:

    kmeans = KMeans(n_clusters=k, random_state=0).fit(newdf[['std','prop_pos_day_change']].as_matrix())

    newdf[str(k)] = kmeans.labels_

    print("Finished k=", k)



newdf = pd.melt(newdf, 

                id_vars=["symbol", 'std', 'prop_pos_day_change'],

                var_name="k", 

                value_name="values",

                value_vars=list(newdf.columns[-7:]))



newdf.head()



g = sns.FacetGrid(newdf, col="k", hue="values", col_wrap=4, palette='Set2')

g = g.map(plt.scatter, "std", "prop_pos_day_change")

g.set(xlabel="Closing Deviation")

g.set(ylabel="'Daily Positive Change (%)")

g.fig.suptitle("Stock Cluster Analysis", size=28)

g.fig.subplots_adjust(top=.8)

plt.subplots_adjust(hspace=1.2, wspace=0.4)

g.add_legend()

g._legend.set_title("Cluster")

#handles = g._legend_data.values()

#labels = g._legend_data.keys()

#g.fig.legend(handles=handles, labels=labels, loc='lower right', ncol=3)

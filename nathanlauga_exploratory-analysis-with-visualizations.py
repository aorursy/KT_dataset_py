import pandas as pd
import dask.dataframe as dd
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col, size=4, aspect=2)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()
    plt.show()

def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, size=4, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()
    plt.show()

def plot_correlation_map(df):
    corr = df.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 12}
    )
    plt.show()

def describe_more(df):
    var = [];
    l = [];
    t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by='Levels', inplace=True)
    return levels
def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['click_hour'] = df['click_time'].apply(lambda x: x.strftime('%H')).astype(int)

    return df

def dataPreProcess(df):
    df = dataPreProcessTime(df)
    df = df.fillna(0)
    return df
path = '../input/'
train = dd.read_csv(path + "train.csv")
freq = 0.02
train = train.random_split([freq, 1-freq], random_state=42)[0]
train = train.compute()

train.columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'attributed_time', 'is_attributed']
train = dataPreProcess(train)
train.head()
train.shape
describe_more(train)
plot_categories( train , cat = 'click_hour' , target = 'is_attributed' )
plot_distribution( train , var = 'os' , target = 'is_attributed' )
plot_distribution( train , var = 'channel' , target = 'is_attributed' )
plot_distribution( train , var = 'app' , target = 'is_attributed' )
plot_correlation_map(train)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy 
import pandas
import matplotlib.pyplot as plt
import seaborn; seaborn.set_style("whitegrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
news_train_df.head()
news_train_df.info(null_counts=True)
(market_train_df.assetCode.value_counts() == 0).sum() # .value_counts()  # .sort_index(ascending=False)
seaborn.distplot(market_train_df.assetCode.value_counts(), kde=False, bins=numpy.arange(0, 2500, 50))
# Take AAPL as a reference in terms of data availability
market_train_df[market_train_df['assetName'] == 'Apple Inc'].head()
def compare_returns(df):
    
    df = df[df['assetName'] == 'Apple Inc']
    df['my_returnsClosePrevRaw1'] = df['close'] / df.shift(1)['close'] - 1
    df['log_returnsClosePrevRaw1'] = numpy.log(df['close'] / df.shift(1)['close']) # - 1
    df['my_returnsOpenPrevRaw1'] = df['open'] / df.shift(1)['open'] - 1
    df['my_returnsOpenPrevRaw10'] = df['open'] / df.shift(10)['open'] - 1
        
    return df[['returnsClosePrevRaw1','my_returnsClosePrevRaw1', 'log_returnsClosePrevRaw1']].head()

compare_returns(market_train_df).iloc[1][0], compare_returns(market_train_df).iloc[1][1], compare_returns(market_train_df).iloc[1][2]
market_train_df.corr()
market_train_df[['returnsOpenNextMktres10']].describe()
news_train_df.urgency.value_counts()
news_train_df.query("urgency == 1")
(news_train_df.assetName.value_counts() > 0).sum()
market_train_df.assetName.unique().isin(news_train_df.assetName.unique()).sum() / market_train_df.assetName.unique().size
market_train_df.query("assetCode == 'AAPL.O'") #.iloc[0]['assetName']
news_train_df.query("(assetName == 'Apple Inc') and (time >= '2007-02-01 22:00:00+00:00')") #.iloc[0]['assetName']
news_train_df.sourceId.value_counts() #("sourceId == 'e58c6279551b85cf'")
news_train_df.query("sourceId == '65b52d8325df3e17'").iloc[0]
news_train_df.urgency.value_counts()
news_train_df.query("urgency == 2")['headlineTag']
news_train_df.query("headlineTag == 'BRIEF'")['urgency'].unique()
news_train_df.assetName.value_counts()[news_train_df.assetName.value_counts() > 0]
market_train_df.assetName.value_counts()[market_train_df.assetName.value_counts() > 0]
# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
market_obs_df.head()
predictions_template_df.head()
market_train_df.universe.value_counts()
market_train_df[market_train_df['universe'] == 1].assetCode.unique().size, market_train_df[market_train_df['universe'] == 0].assetCode.unique().size
#market_train_df.assign(universe_ratio = lambda x: )
seaborn.distplot(market_train_df.groupby('assetCode')['universe'].apply(lambda x: (x==1).sum() / x.size), kde=False, bins=numpy.arange(0,1.01,0.05))
def describe_no_outliers(df):
    upper_bound_in = numpy.quantile(df.query('universe == 1')['returnsOpenNextMktres10'], 0.99)
    lower_bound_in = numpy.quantile(df.query('universe == 1')['returnsOpenNextMktres10'], 0.01)
    upper_bound_out = numpy.quantile(df.query('universe == 0')['returnsOpenNextMktres10'], 0.99)
    lower_bound_out = numpy.quantile(df.query('universe == 0')['returnsOpenNextMktres10'], 0.01)
    
    return pandas.DataFrame([
         df[df['returnsOpenNextMktres10'].between(lower_bound_in, upper_bound_in)].query('universe == 1').describe()['returnsOpenNextMktres10'], 
         df[df['returnsOpenNextMktres10'].between(lower_bound_out, upper_bound_out)].query('universe == 0').describe()['returnsOpenNextMktres10'], 
        ])

describe_no_outliers(market_train_df)
fig, ax = plt.subplots(ncols=2, figsize=(10,3), sharey='row')
market_train_df.query('universe == 0')['returnsOpenNextMktres10'].plot.hist(grid=True, ax=ax[0], bins=numpy.arange(-0.5,0.5,0.01), label='out of universe')
market_train_df.query('universe == 1')['returnsOpenNextMktres10'].plot.hist(grid=True, ax=ax[1], bins=numpy.arange(-0.5,0.5,0.01), label='in universe')
ax[0].legend(); ax[1].legend();
import sklearn.metrics

def results_baseline(df):
    
    print(df[df['returnsOpenNextMktres10'] == 0].index.size)
    df = df.assign(
        close_open = df['close'] / df['open'] - 1,
        prev_10 = df['returnsOpenPrevMktres10'],
    )
    
    def metrics(pred):
        true = df['returnsOpenNextMktres10'] > 0
        pred = df[pred] > 0 
        tot = df.index.size
        tn, fp, fn, tp = sklearn.metrics.confusion_matrix(true, pred).ravel()
        
        return pandas.Series({
            'accuracy': sklearn.metrics.accuracy_score(true, pred),
            'up_precision': sklearn.metrics.precision_score(true, pred),
            'prevalence': true.mean(),
            'down_precision': sklearn.metrics.precision_score(~true,~pred),
            'inv_prevalence': (~true).mean(),
            'TN': tn/tot, 'FP': fp/tot, 'FN':fn/tot, 'TP':tp/tot
        })
    
    return pandas.DataFrame({
        'close_open': metrics('close_open'),
        'prev_10': metrics('prev_10')
    })

results_baseline(market_train_df)
pandas.DataFrame({'out_pearson': market_train_df.query('universe == 0').corr()['returnsOpenNextMktres10'],
                  'in_pearson': market_train_df.query('universe == 1').corr()['returnsOpenNextMktres10'],
                 }).sort_values(by='out_pearson')

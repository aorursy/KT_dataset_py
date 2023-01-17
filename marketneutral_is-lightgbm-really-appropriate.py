import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['figure.figsize'] = 14, 8
# Make environment and get data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
from scipy.stats import spearmanr

def sp(group, col1_name, col2_name):
    x = group[col1_name]
    y = group[col2_name]
    return spearmanr(x, y)[0]
market_train_df['target_shift_1'] = (
    market_train_df.
    groupby('assetCode')['returnsOpenNextMktres10'].
    shift(-1)
)

market_train_df['target_shift_5'] = (
    market_train_df.
    groupby('assetCode')['returnsOpenNextMktres10'].
    shift(-5)
)
rc_1 = (
    market_train_df.
    sort_values(['time', 'assetCode']).
    dropna().
    groupby('time').
    apply(sp, 'returnsOpenNextMktres10', 'target_shift_1')
)

rc_5 = (
    market_train_df.
    sort_values(['time', 'assetCode']).
    dropna().
    groupby('time').
    apply(sp, 'returnsOpenNextMktres10', 'target_shift_5')
)
rc_1.plot()
rc_5.plot(title='Rank Correlation Between Target and Target Shifted: 1 Day and 5 Days');
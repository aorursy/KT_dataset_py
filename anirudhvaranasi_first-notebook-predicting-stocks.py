# Creating Environment 
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print("Data Loaded")
market_train_df, news_train_df = env.get_training_data()
print("We have {:,} market samples and {} features in the training dataset.".format(market_train_df.shape[0], market_train_df.shape[1]))
print("We have {:,} news samples and {} features in the training dataset.".format(news_train_df.shape[0], news_train_df.shape[1]) )
market_train_df.head()
news_train_df.head()
# What do variables look like?
market_train_df.describe()
market_train_df.describe(include=['O'])
news_train_df.describe()
news_train_df.describe(include=['O'])
# import visulization packages
import matplotlib.pyplot as plt

# set up the figure size
%matplotlib inline
plt.rcParams['figure.figsize'] = (20, 30)

# make subplots
fig, axes = plt.subplots(nrows = 6, ncols = 2)

# Specify the features of interest
num_features = ['volume', 'close', 'open','returnsClosePrevRaw1','returnsOpenPrevRaw1','returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
                'returnsClosePrevRaw10','returnsOpenPrevRaw10','returnsClosePrevMktres10', 'returnsOpenPrevMktres10','returnsOpenNextMktres10']
xaxes = num_features
yaxes = ['Counts', 'Counts', 'Counts','Counts', 'Counts','Counts', 'Counts','Counts', 'Counts','Counts', 'Counts','Counts']

# draw histograms
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(market_train_df[num_features[idx]].dropna(), bins=40)
    ax.set_xlabel(xaxes[idx], fontsize=20)
    ax.set_ylabel(yaxes[idx], fontsize=20)
    ax.tick_params(axis='both', labelsize=15)
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
sns.set(color_codes=True)
sns.distplot(market_train_df['volume'], hist=False, rug=True);

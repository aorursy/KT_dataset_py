import pandas as pd
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import json
import pylab as plt
from scipy.stats.mstats import zscore
combined_df = pd.read_csv('../input/data-cleaning/combined_measures.csv', index_col='date').dropna()
sns.pairplot(combined_df[['tweets', 'twitter_mentions', 'mood_baseline', 'mood_cat']], hue="mood_cat")
from scipy import stats
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
sns.jointplot(y="tweets", x="twitter_mentions", data=combined_df, kind='reg', stat_func=r2)
sns.boxplot(y="tweets", x="mood_cat", data=combined_df,
            whis="range")
X = combined_df[['tweets', 'mood_baseline']]
X = sm.add_constant(X)
y = combined_df['mood_int']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
sns.regplot(y="tweets", x="mood_int", data=combined_df, x_estimator=np.mean)
sns.boxplot(y="twitter_mentions", x="mood_cat", data=combined_df,
            whis="range")
X = combined_df[['twitter_mentions', 'mood_baseline']]
X = sm.add_constant(X)
y = combined_df['mood_int']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()
sns.regplot(y="twitter_mentions", x="mood_int", data=combined_df, x_estimator=np.mean)
X = combined_df[['twitter_mentions', 'tweets', 'mood_baseline']]
X = sm.add_constant(X)
y = combined_df['mood_int']

# Note the difference in argument order
model = sm.OLS(y, X).fit()
predictions = model.predict(X) # make the predictions by the model

# Print out the statistics
model.summary()

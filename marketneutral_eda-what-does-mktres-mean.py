import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 7
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
del news_train_df
df = (
    market_train_df.
    reset_index().
    sort_values(['assetCode', 'time']).
    set_index(['assetCode','time'])
)

df['implied_mkt_return'] = (
    df.
    groupby('assetCode').
    apply(lambda x: x.returnsClosePrevRaw1 - x.returnsClosePrevMktres1).
    reset_index(0, drop=True)
)
plt.scatter(
    df.loc['AAPL.O'].implied_mkt_return,
    df.loc['NFLX.O'].implied_mkt_return,
    alpha=0.6
);
plt.xlabel('Implied Market Return from AAPL');
plt.ylabel('Implied Market Return from NFLX');
returnsClosePrevRaw1 = (
    df['returnsClosePrevRaw1'].
    swaplevel().
    unstack()
)

returnsClosePrevMktres1 = (
    df['returnsClosePrevMktres1'].
    swaplevel().
    unstack()
)
num_days = 260*5  # Take 5 years
num_stocks = 200  # Try for 200 stocks but we will get many less due to NaNs

returnsClosePrevRaw1 = \
    returnsClosePrevRaw1.iloc[-num_days:, 0:num_stocks].dropna(axis=1)
num_stocks = len(returnsClosePrevRaw1.columns)
print(num_stocks)
returnsClosePrevMktres1 = (
    returnsClosePrevMktres1.
    loc[returnsClosePrevRaw1.index][returnsClosePrevRaw1.columns].
    clip(lower=-0.15, upper=0.15)
)
from sklearn.decomposition import PCA
pca = PCA(n_components=15, svd_solver='full')
pca.fit(returnsClosePrevRaw1)
plt.bar(range(15),pca.explained_variance_ratio_);
plt.title('Principal Components Sorted by Variance Explain');
plt.ylabel('% of Total Variance Explained');
plt.xlabel('PC factor number');
pcs = pca.transform(returnsClosePrevRaw1)
# It's always tricky keeping the dimensions right, so I am going to print them for reference.
print(num_stocks)
print(num_days)
print(np.shape(pca.components_))
print(np.shape(pcs))
# the market return is the first PC
mkt_return = pcs[:,0].reshape(num_days,1)

# the betas of each stock to the market return are in
# the first column of the components
mkt_beta = pca.components_[0,:].reshape(num_stocks,1)

# the market portion of returns is the projection of one onto the other
mkt_portion = mkt_beta.dot(mkt_return.T).T

# ...and the residual is just the difference
residual = returnsClosePrevRaw1 - mkt_portion
print(mkt_return.shape)
print(mkt_portion.shape)
from sklearn.covariance import LedoitWolf

def get_corr_from_cov(covmat):
    d = np.diag(np.sqrt(np.diag(lw.covariance_)))
    return np.linalg.inv(d).dot(lw.covariance_).dot(np.linalg.inv(d))

lw = LedoitWolf()

lw.fit(returnsClosePrevMktres1)
corr = get_corr_from_cov(lw.covariance_)

lw.fit(residual)
corr2 = get_corr_from_cov(lw.covariance_)
from scipy.spatial import distance
from scipy.cluster import hierarchy

def plot_side_by_side_hm(corr, corr2, title1, title2):
    row_linkage = hierarchy.linkage(
        distance.pdist(corr), method='average')
    row_order = list(map(int, hierarchy.dendrogram(row_linkage, no_plot=True)['ivl']))
    
    col_linkage = hierarchy.linkage(
        distance.pdist(corr.T), method='average')
    col_order = list(map(int, hierarchy.dendrogram(col_linkage, no_plot=True)['ivl']))
    
    corr_swapped = np.copy(corr)
    corr_swapped[:, :] = corr_swapped[row_order, :]
    corr_swapped[:, :] = corr_swapped[:, col_order]

    corr_swapped2 = np.copy(corr2)
    corr_swapped2[:, :] = corr_swapped2[row_order, :]
    corr_swapped2[:, :] = corr_swapped2[:, col_order]

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    fig.tight_layout()
    cs1 = sns.heatmap(corr_swapped, square=True, xticklabels=False, yticklabels=False, cbar=False, ax=ax1, cmap='OrRd')
    cs1.set_title(title1)
    cs2 = sns.heatmap(corr_swapped2, square=True, xticklabels=False, yticklabels=False, cbar=False, ax=ax2, cmap='OrRd')
    cs2.set_title(title2);

plot_side_by_side_hm(
    corr,
    corr2,
    'Hierarchical Correlation Matrix: returnsClosePrevMktres1',
    'Correlation Matrix: Our Residual Est (mapped to <-- hierarchy)'
)
lw.fit(returnsClosePrevRaw1)
corr = get_corr_from_cov(lw.covariance_)

lw.fit(returnsClosePrevMktres1)
corr2 = get_corr_from_cov(lw.covariance_)
plot_side_by_side_hm(
    corr,
    corr2,
    'Hierarchical Correlation Matrix: returnsClosePrevRaw1',
    'Correlation Matrix: returnsClosePrevMktres1 (mapped to <-- hierarchy)'
)
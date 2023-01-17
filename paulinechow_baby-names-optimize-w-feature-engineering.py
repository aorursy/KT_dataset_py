#Load necessary libraries for code
import numpy as np 
import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.signal #peak analysis 

# Supress unnecessary warnings so that presentation looks clean
import warnings
warnings.filterwarnings("ignore")
import datetime

import bq_helper
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
color = sns.color_palette()

from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# set number of rows in print statements
# helps with online notebooks
pd.options.display.max_rows = 1000
# create a helper object for this dataset
usa_names = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="usa_names")

# query and export data 
query = """SELECT year, gender, name, sum(number) as number FROM `bigquery-public-data.usa_names.usa_1910_current` GROUP BY year, gender, name"""
agg_names = usa_names.query_to_pandas_safe(query)
agg_names.to_csv("usa_names.csv")
#outputing data according to a certain organization
agg_names.sort_values(["gender", "name", "year"]).reset_index(drop=True, inplace=True)
agg_names.head()
#shape and size of dataset
agg_names.shape
#get full list of years, check if there are incomplete years
np.sort(agg_names.year.unique())
# max and min of years in the dataset
print("year min and max", agg_names.year.min(), agg_names.year.max())
all_years = list(range(agg_names.year.min(), agg_names.year.max()))
# check for categories 
agg_names.gender.unique()
#total female population for all years
# Surprise gender!? Then no need to filter dataframe for females
df_female = agg_names[agg_names.gender == 'F']
df_female_pivot = df_female.pivot(index='year', columns='name', values='number')
df_female_pivot.fillna(0, inplace=True)
df_female_total = df_female_pivot.sum(axis=1)
df_female_total_list = list(df_female.name.unique())
# graphs for post
#print(sns.__version__) #0.8.1
#sns.lineplot(data=df_female_pivot.loc[:, "Tiana"], palette="tab10", linewidth=2.5)

princess_names = ["Tiana", "Ariel", "Aurora", "Elsa"]
movie_years = [2009, 2000, 1959, 2013]

ax = df_female_pivot.loc[:, princess_names].plot(
    title="Babies with \n Disney Princess Names \n with Movie Release Years \n (SSA Data)"
    , grid=True, legend=True)
ax.set_ylabel("Number of Babies")

#note release of corresponding Disney Movie with Name
for pi in range(0, len(princess_names)): 
    plt.plot(movie_years[pi], 
             np.max(df_female_pivot.loc[:, princess_names[pi]]), '+', color='black', marker='o')                 

def plot_peaks(name, x, indexes, algorithm=None, mph=None, mpd=None):
    """Helper function 
    Plot results of the peak dectection.
    Function from https://github.com/MonsieurV/py-findpeaks/blob/master/tests/vector.py"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
        return
    _, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(x, 'b', lw=1)
    if indexes.size:
        label = 'peak'
        label = label + 's' if indexes.size > 1 else label
        ax.plot(indexes, x[indexes], '+', mfc=None, mec='r', mew=2, ms=8,
                label='%d %s' % (indexes.size, label))
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.02*x.size, x.size*1.02-1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
    ax.set_xlabel('Data #', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    ax.set_title('%s %s (mph=%s, mpd=%s)' % (name, algorithm, mph, mpd))
plt.show()
def peak_detection_simple(trend):
    '''
    simple approach for demonstration
    compute differences between consecutive time periods and count the number of sign changes
    returns the number of sign changes, only if it is from positive to negative 
    '''
    ch = [x1-x0 for x0,x1 in zip(trend,trend[1:]) if x1 != x0]
    return sum([1 if c < 0 else 0 for c in ch])
test_name = "Clarissa"
print('number of sign changes from positive to negative: {}'.format(peak_detection_simple(df_female_pivot['Clarissa'])))
def get_peaks(df, name, d, verbose=False):
    '''
    Function takes a dataframe, name (string), d (distance of peak)
    Returns index or years of applicable time frames
    '''
    df_filter = df.loc[:, name]
    tvec = np.array(df_filter) 
    indexes, _ = scipy.signal.find_peaks(tvec, height=float(tvec.mean()), distance=d)
    if verbose: 
        print('Peaks are: %s' % (indexes))
        print(tvec[indexes])
    return indexes
ind = get_peaks(df_female_pivot, test_name, d=5)
print('Print output of indicies with peaks for {}: {}'.format(test_name, ind))
c=0
for i in range(1, len(ind)): 
    if ind[i] - ind[i-1] <= 12: 
        #print(indexes[i] - indexes[i-1])
        c+=1

#calc the range of min and max peaks -- important if within 16 years of each other 
print('Print output of indicies: {}'.format(c))
print('''Is difference between last and first years <= 16 years: {}'''.format(ind[-1] - ind[0] <= 16))
plot_peaks(test_name, 
    np.array(df_female_pivot.loc[:, test_name]),
    ind, mph=0, mpd=0, algorithm='scipy.signal.find_peaks'
)
def yoy_calc(s, df):
    return df.apply(lambda x: (x - x.shift(s)) / x, axis=0)

yoy_female = yoy_calc(1, df_female_pivot)
yoy5_female = yoy_calc(5, df_female_pivot)
#top 500 name within the last 3 years gets filtered out

now = datetime.datetime.now()
last_3_yrs = list(range(now.year - 3,now.year))
count_3_yrs = df_female_pivot.loc[last_3_yrs, :]
df_3_yrs = pd.DataFrame(count_3_yrs.sum().reset_index())
df_3_yrs.columns= ['names', 'count']
df_3_yrs.set_index('names', inplace=True)
df_3_yrs['rank_3yr'] = df_3_yrs.rank(ascending=False)
#let's sample some of the top 500 names 
top_500_list = df_3_yrs[df_3_yrs.loc[:, 'rank_3yr'] <= 500].index
#df_3_yrs[df_3_yrs.loc[:, 'rank_3yr'] <= 500].sort_values('rank_3yr').head(20)

#dataframe of girls name, not including top 500 
df_not_top = df_3_yrs[df_3_yrs.loc[:, 'rank_3yr'] > 500]
#create fresh dataframe with female names in index
df_metrics = pd.DataFrame(index=list(df_female_pivot.columns))
last_yr = df_female_pivot.index[-1]
#np.array(df_female_pivot.loc[:, 'Ema'])
np.array(df_female_total)
for n in df_female_pivot.columns: 
    ipeaks = get_peaks(df_female_pivot, n, d=5)
    #print(n, ipeaks)
    
    if ipeaks.any(): 
    
        #acceleration of names over x years 
        df_metrics.loc[n, "acc_last_1_yr"] = np.mean(yoy_female.loc[(last_yr):, n])
        df_metrics.loc[n, "acc_last_2_yr"] = np.mean(yoy_female.loc[(last_yr-2):, n])
        df_metrics.loc[n, "acc_last_5_yr"] = np.mean(yoy_female.loc[(last_yr-5):, n])
        df_metrics.loc[n, "acc_last_10_yr"] = np.mean(yoy_female.loc[(last_yr-10):, n])
        df_metrics.loc[n, "acc_last_15_yr"] = np.mean(yoy_female.loc[(last_yr-15):, n])
        df_metrics.loc[n, "acc_last_20_yr"] = np.mean(yoy_female.loc[(last_yr-20):, n])
        df_metrics.loc[n, "acc_last_25_yr"] = np.mean(yoy_female.loc[(last_yr-25):, n])

        #acceleration within the last x years
        df_metrics.loc[n, "acc_5_yr"] = np.divide((df_female_pivot.loc[last_yr, n] - df_female_pivot.loc[last_yr -5, n]), df_female_pivot.loc[last_yr -5, n])
        df_metrics.loc[n, "acc_10_yr"] = np.divide((df_female_pivot.loc[last_yr, n] - df_female_pivot.loc[last_yr -10, n]), df_female_pivot.loc[last_yr -10, n])
        df_metrics.loc[n, "acc_15_yr"] = np.divide((df_female_pivot.loc[last_yr, n] - df_female_pivot.loc[last_yr -15, n]), df_female_pivot.loc[last_yr -15, n])
        df_metrics.loc[n, "acc_20_yr"] = np.divide((df_female_pivot.loc[last_yr, n] - df_female_pivot.loc[last_yr -20, n]), df_female_pivot.loc[last_yr -20, n])
        df_metrics.loc[n, "acc_20_yr"] = np.divide((df_female_pivot.loc[last_yr, n] - df_female_pivot.loc[last_yr -25, n]), df_female_pivot.loc[last_yr -25, n])

        #add peaks metrics to the dataframe
        peak_pop = np.array(df_female_pivot.loc[:, n])[ipeaks]
        female_pop = np.array(df_female_total)[ipeaks]
        peaks_since = [len(df_female_pivot.index) - p for p in ipeaks] #how many years since peak? 

        df_metrics.loc[n, "peak_count"] = len(ipeaks)
        df_metrics.loc[n, "peak_recent_years_since"] = min(peaks_since)
        df_metrics.loc[n, "peaks_within_5_years"] = 1 if ipeaks[-1] - ipeaks[0] < 6 else 0

        #calculate avg, min, max, median population at peaks
        pop_perc = list(map(lambda x, y: np.divide(x,y), peak_pop,female_pop))
        df_metrics.loc[n, "peaks_avg_pop_perc"] = np.mean(pop_perc)
        df_metrics.loc[n, "peaks_median_pop_perc"] = np.median(pop_perc)
        df_metrics.loc[n, "peaks_min_pop_perc"] = min(pop_perc)
        df_metrics.loc[n, "peaks_max_pop_perc"] = max(pop_perc)

        #indicators for x years since peaks
        for pp in peaks_since: 
            if pp <= 5: 
                df_metrics.loc[n, "peak_last_5_years"] = 1
            elif pp <= 10: 
                df_metrics.loc[n, "peak_last_10_years"] = 1
            elif pp < 15:
                df_metrics.loc[n, "peak_last_15_years"] = 1
            elif pp < 20: 
                df_metrics.loc[n, "peak_last_20_years"] = 1
            elif pp < 25: 
                df_metrics.loc[n, "peak_last_25_years"] = 1

        #Ever in top y names in the past x years?
        df_metrics['top_3_yrs'] = 1 if n in top_500_list else 0

#check outputs
df_metrics.replace([np.inf, -np.inf], np.nan, inplace=True)
df_metrics.fillna(0, inplace=True)
df_metrics.head()
df_metrics.shape
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
cats = ["peaks_within_5_years", "peak_last_15_years", "peak_last_5_years", "top_3_yrs",
        "peak_last_10_years", "peak_last_25_years", "peak_last_20_years"]
use_cols = [d for d in df_metrics.columns if d not in cats]
x_nums = df_metrics.loc[:, use_cols]
x_scale_num = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(x_nums)
df_scale = pd.DataFrame(data=x_scale_num, index=list(df_metrics.index), columns=use_cols)
x_merge = df_scale.join(df_metrics.loc[:, cats])
X = np.array(x_merge.as_matrix())
def get_cluster_score(data): 
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n).fit(data)
        label = kmeans.labels_
        sil_coeff = silhouette_score(data, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n, sil_coeff))
get_cluster_score(X)
def get_labeled_clusters(c, data, merged,name_ls, desired_name):
    tmp = KMeans(n_clusters=c).fit(data).labels_
    k4 = pd.DataFrame(tmp, index=name_ls, columns=["labels"])
    k4_labels = pd.concat([k4, merged], axis=1)
    print(k4_labels.loc[:, ["labels"]].reset_index().groupby("labels").count())
    print(desired_name, ' is in ', k4_labels.loc[desired_name, "labels"])
    
    return (k4_labels, k4_labels.loc[desired_name, "labels"])
df_r, p_label = get_labeled_clusters(10, X, x_merge, list(df_metrics.index), "Pauline")
#list of names from cluster of desired name
r = list(df_r[df_r.labels==p_label].index)
print(r)
df_r[df_r.labels==p_label].to_csv("baby_names_metrics_cluster.csv")
ax = df_female_pivot.loc[:, r].plot(
    title="Baby Names \n Similar to 'Pauline' \n Clustering with Metrics\n (SSA Data)"
    , grid=True, legend=False)
ax.set_ylabel("Number of Babies")

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
tmp = df_female.pivot(index='name', columns='year', values='number')
tmp.fillna(0, inplace=True)
tmp.shape
sc = StandardScaler()
X_pca_name_ls = list(tmp.index)
X_pca = np.array(sc.fit_transform(tmp))
X_pca.shape
pca_check = PCA(n_components=25)
pca_check.fit(X_pca)
print(np.cumsum(pca_check.explained_variance_ratio_))

plt.plot(np.cumsum(pca_check.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.title('Name Trends PCA')
pca = PCA(n_components=10)
X_pca_transform = pca.fit_transform(X_pca)
X_pca_transform.shape
get_cluster_score(X_pca_transform)
#get_labeled_clusters(4, X_pca_transform, df_female, X_pca_name_ls, "Pauline")
tmp = KMeans(n_clusters=10).fit(X_pca_transform).labels_
data_components = pd.DataFrame(X_pca_transform, index=X_pca_name_ls)
data_components['labels'] = tmp
data_components.head()
print(data_components.loc[:, ["labels"]].reset_index().groupby("labels").count())
p_label2 = data_components.loc["Pauline", "labels"]
print(p_label2)
#list of names from cluster 2
r2 = list(data_components[data_components.labels==p_label2].index)
print(r2)
data_components[data_components.labels==p_label2].to_csv("baby_names_metrics_cluster.csv")
ax = df_female_pivot.loc[:, r2].plot(
    title="Baby Names \n Similar to 'Pauline' \n Clustering with PCA\n (SSA Data)"
    , grid=True, legend=False)
ax.set_ylabel("Number of Babies")
# combine X_pca_transform and X for cluster analysis
# number of clusters 
X_combo = np.concatenate((X,X_pca_transform),axis=1)
get_cluster_score(X_combo)
data_combo, p_label3 = get_labeled_clusters(6, X_combo, df_female, X_pca_name_ls, "Pauline")
r3 = list(data_components[data_components.labels==p_label3].index)
print(r3)
data_combo[data_combo.labels==p_label3].to_csv("baby_names_combo_cluster.csv")
ax = df_female_pivot.loc[:, r3].plot(
    title="Baby Names \n Similar to 'Pauline' \n Clustering with Combined Feaures\n (SSA Data)"
    , grid=True, legend=False)
ax.set_ylabel("Number of Babies")
both = set(r).intersection(set(r2))
print(both)
print(len(both))
set(r).intersection(set(r3))
print(len(r))
print(len(r2))
#import random
#random.sample(r3, 10)

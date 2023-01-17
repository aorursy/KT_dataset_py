import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pylab as plt
train = pd.read_csv('../input/train.csv', parse_dates=[0])
test = pd.read_csv('../input/test.csv', parse_dates=[0])
submit = pd.read_csv('../input/sample_submission.csv')
plt.style.use('ggplot')
fake_store_names = {1: 'Albertsons', 2: 'Ahold', 3: 'Food_Lion', 4: 'Hannaford',
                    5: 'Giant', 6: 'Stop_n_Shop', 7: 'Kroger', 8: 'SpartanNash',
                    9: 'SuperValu', 10: 'Walmart'}

fake_items = {1:'Apples', 2:'Bacon', 3:'Bagles', 4:'Beans', 5:'Beer', 6:'Bread',
              7:'Carrots', 8:'Cheese', 9:'Chips', 10:'Coffee', 11:'Cream', 
              12:'Egg', 13:'Fish', 14:'Foil', 15:'Granola Bars', 16:'Grapes',
              17:'Ham', 18:'Honey', 19:'Ice Cream', 20:'Ketchup', 21:'Kielbasa',
              22:'Lemons', 23:'Lettuce', 24:'Margarine', 25:'Mayonnaise', 26:'Milk',
              27:'Mushrooms', 28:'Mustard', 29:'Oranges', 30:'Paper Towles',
              31:'Pasta', 32:'Peanut Butter', 33:'Pears', 34:'Pizza', 35:'Plastic Wrap',
              36:'Potatoes', 37:'Pretzels', 38:'Ribs', 39:'Rice', 40:'Salami', 
              41:'Salsa', 42:'Salt', 43:'Sausage', 44:'Soda', 45:'Soup', 46:'Sugar',
              47:'Tuna', 48:'Turkey', 49:'Waffles', 50:'Yoghurt'}

train['store_name'] = train['store'].replace(fake_store_names)
train['item_name'] = train['item'].replace(fake_items)
grouped = train.groupby(by=['item_name'])
for i, d in grouped:
    myplot = d.set_index('date').groupby('store_name')['sales'] \
        .plot(figsize=(15,2), style='.', title=str(i), legend=False)
    plt.show()
def plot_year_over_year(item, store):
    sample = train.loc[(train['store'] == store) & (train['item'] == item)].set_index('date')
    pv = pd.pivot_table(sample, index=sample.index.month, columns=sample.index.year,
                        values='sales', aggfunc='sum')
    ax = pv.plot(figsize=(15,3), title=fake_store_names[store] + ' - ' + fake_items[item])
    ax.set_xlabel("Month")
plot_year_over_year(1, 1)
plot_year_over_year(1, 2)
plot_year_over_year(20, 5)
plot_year_over_year(20, 6)
def plot_year_over_year_dow(item, store):
    sample = train.loc[(train['store'] == store) & (train['item'] == item)].set_index('date')
    pv = pd.pivot_table(sample, index=sample.index.weekday, columns=sample.index.year,
                        values='sales', aggfunc='sum')
    ax = pv.plot(figsize=(15,3), title=fake_store_names[store] + ' - ' + fake_items[item])
    ax.set_xlabel("Day of Week")
plot_year_over_year_dow(1, 1)
plot_year_over_year_dow(1, 2)
plot_year_over_year_dow(20, 5)
plot_year_over_year_dow(20, 6)
# Data prep
train['store_item'] = train['store_name'] + '-' + train['item_name']
train['store_item_mean'] = train.groupby('store_item')['sales'].transform('mean')
train['deviation_from_storeitem_mean'] = train['sales'] - train['store_item_mean']
train['dev_rolling'] = train.groupby('store_item')['deviation_from_storeitem_mean'].rolling(30).mean().reset_index()['deviation_from_storeitem_mean']
train_pivoted = train.pivot(index='store_item', columns='date', values='sales')
train_pivoted.head()
deviation_pivot = train.pivot(index='store_item', columns='date', values='dev_rolling')
deviation_pivot = deviation_pivot.dropna(axis=1)
deviation_pivot.head()
# Example from here:
# https://stackoverflow.com/questions/34940808/hierarchical-clustering-of-time-series-in-python-scipy-numpy-pandas
    
import scipy.cluster.hierarchy as hac
from scipy import stats
# Here we use spearman correlation
def my_metric(x, y):
    r = stats.pearsonr(x, y)[0]
    return 1 - r # correlation to distance: range 0 to 2

Z = hac.linkage(deviation_pivot, method='single', metric=my_metric)
from scipy.cluster.hierarchy import fcluster

def print_clusters(deviation_pivot, Z, k, plot=False):
    # k Number of clusters I'd like to extract
    results = fcluster(Z, k, criterion='maxclust')

    # check the results
    s = pd.Series(results)
    clusters = s.unique()

    for c in clusters:
        cluster_indeces = s[s==c].index
        print("Cluster %d number of entries %d" % (c, len(cluster_indeces)))
        if plot:
            deviation_pivot.T.iloc[:,cluster_indeces].plot()
            plt.show()

print_clusters(deviation_pivot, Z, 5, plot=False)
from sklearn.cluster import KMeans
clust = KMeans()
deviation_pivot['cluster'] = clust.fit_predict(deviation_pivot)
deviation_pivot.T

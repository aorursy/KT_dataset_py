# the required imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from tqdm import tqdm
import gc
input_dir = '../input'

train_df = pd.read_csv(input_dir + '/train.csv')
train_df = train_df.drop(['ID', 'target'], axis = 1)
test_df = pd.read_csv(input_dir + '/test.csv')
test_df = test_df.drop(['ID'], axis = 1)
features = pd.concat([train_df, test_df], ignore_index=True)


data = np.log1p(features)

scaler = StandardScaler(with_mean=False)
data = scaler.fit_transform(data)
scaled_features = pd.DataFrame(data = data, columns=features.columns)

del train_df, test_df, data, features
gc.collect()
bins=300

features = scaled_features.columns
ranges = np.linspace(np.min(np.min(scaled_features,axis=0)), np.max(np.max(scaled_features,axis=0)), bins+1)

feat_hist_df =  pd.DataFrame(columns=ranges[:-1])

for feat in tqdm(features, ncols=110):
    hist = pd.DataFrame(np.histogram(scaled_features[feat], bins=ranges)[0]\
                        .reshape(1,-1), columns=ranges[:-1], index= [feat])
    feat_hist_df = feat_hist_df.append(hist)
feat_hist_nozero_df = feat_hist_df.drop([0],axis=1)
af = AffinityPropagation().fit(feat_hist_nozero_df)
feat_hist_nozero_df['cluster'] = af.labels_
print('Using Affinity Propagation resulted in total of : {} clusters'.format(len(af.cluster_centers_indices_)))
N = 10
cluster_count = np.unique(af.labels_, return_counts=True)
# as a sorted DataFrame
cluster_count = pd.DataFrame({'cluster':cluster_count[0],'count':cluster_count[1]})\
                  .sort_values(by=['count'], ascending=False).reset_index(drop = True)
# obtaining the top and bottom clusters
top_clusters = cluster_count.head(N)['cluster'].tolist()
top_counts = cluster_count.head(N)['cluster'].sum()
bottom_clusters = cluster_count.tail(N)['cluster'].tolist()
top_accounts_percent = np.around(100 * top_counts / len(features),2)
print('Top {} cluster(s) accounts for {}% of total features'.format(N,top_accounts_percent))
print(cluster_count.head(N).T)
print('')
print('Bottom {} cluster(s)'.format(N))
print(cluster_count.tail(N).T)

# the data will be converted into "long" format so it could be visualize using sns.tsplot
feat_hist_nozero_long = feat_hist_nozero_df.reset_index().melt(id_vars=['index','cluster'],
                                                               var_name='bins', value_name='count')
intop = np.in1d(feat_hist_nozero_long.cluster, top_clusters)
inbottom = np.in1d(feat_hist_nozero_long.cluster, bottom_clusters)
plt.figure(figsize=(16,8))
sns.tsplot(data=feat_hist_nozero_long[intop], time='bins',value='count',unit='index',condition ='cluster')
plt.title('Histograms for top N cluster(s)')
plt.grid()
plt.figure(figsize=(16,8))
sns.tsplot(data=feat_hist_nozero_long[inbottom], time='bins',value='count',unit='index',condition ='cluster')
plt.title('Histograms for bottom N cluster(s)')
plt.grid()
zeros_count = feat_hist_df[[0]].astype(int)
zeros_count.columns = ['zero_count']
zeros_count['cluster'] = af.labels_
zeros_count = zeros_count.reset_index(drop=True).groupby('cluster')\
                         .agg([np.size, np.mean])['zero_count'].sort_values('mean')
    
x = zeros_count['mean'].values
y1 = zeros_count['size'].values
y2 = np.cumsum(y1)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize = (16,8))
plt.setp((ax1, ax2),xticks=np.arange(0,55000,500))
fig.tight_layout()
ax1.scatter(x, y1, alpha = 0.2)
ax1.grid()
ax1.set_title('Features per Cluster (y) vs. number of zeros per feature (x)')
ax1.set_ylabel('features per cluster')
ax2.plot(x,y2, c = 'g')
ax2.set_xlabel('Mean number of zeros per feature')
ax2.set_ylabel('features')
ax2.set_title('Cumulative Features per Cluster (y) vs. number of zeros per feature (x)')
ax2.grid()
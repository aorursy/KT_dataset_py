import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
df = pd.read_csv("https://raw.githubusercontent.com/noahgift/socialpowernba/master/data/nba_2017_players_with_salary_wiki_twitter.csv");df.head()
salary_pie_social_df = df.loc[:,['PLAYER','SALARY_MILLIONS','PIE','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']];salary_pie_social_df.head()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"));sns.pairplot(salary_pie_social_df, hue="PLAYER")
corr = salary_pie_social_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
corr
plt.figure(figsize=[16,12])

Y = salary_pie_social_df['SALARY_MILLIONS']

plt.subplot(221)
plt.scatter(salary_pie_social_df['PIE'], Y, marker=">")
plt.title('PIE vs SALARY')
plt.xlabel('PIE')
plt.ylabel('SALARY_MILLIONS')


plt.subplot(222)
plt.scatter(salary_pie_social_df['PAGEVIEWS'], Y, marker=">")
plt.title('WIKI_PAGEVIEWS vs SALARY')
plt.xlabel('WIKI_PAGEVIEWS')
plt.ylabel('SALARY_MILLIONS')

plt.subplot(223)
plt.scatter(salary_pie_social_df['TWITTER_FAVORITE_COUNT'], Y, marker=">")
plt.title('TWITTER_FAVORITE_COUNT vs SALARY')
plt.xlabel('TWITTER_FAVORITE_COUNT')
plt.ylabel('SALARY_MILLIONS')

plt.subplot(224)
plt.scatter(salary_pie_social_df['TWITTER_RETWEET_COUNT'], Y, marker=">")
plt.title('TWITTER_RETWEET_COUNT vs SALARY')
plt.xlabel('TWITTER_FAVORITE_COUNT')
plt.ylabel('SALARY_MILLIONS')


results = smf.ols('SALARY_MILLIONS ~PIE', data=salary_pie_social_df).fit()
print(results.summary())
salary_pie_social_predictions_df = salary_pie_social_df.copy()
salary_pie_social_predictions_df["predicted"] = results.predict()
salary_pie_social_predictions_df
sns.lmplot(x="predicted", y="SALARY_MILLIONS", data=salary_pie_social_predictions_df)
numerical_df = salary_pie_social_df.loc[:,['SALARY_MILLIONS','PIE','PAGEVIEWS','TWITTER_FAVORITE_COUNT','TWITTER_RETWEET_COUNT']]
numerical_df.info()
numerical_df['TWITTER_FAVORITE_COUNT'].fillna(numerical_df['TWITTER_FAVORITE_COUNT'].median(), inplace = True)
numerical_df['TWITTER_RETWEET_COUNT'].fillna(numerical_df['TWITTER_RETWEET_COUNT'].median(), inplace = True)

numerical_df.info()
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(numerical_df))
print(scaler.transform(numerical_df))
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3)
kmeans = k_means.fit(scaler.transform(numerical_df))
salary_pie_social_df['cluster'] = kmeans.labels_
salary_pie_social_df.head()
import ggplot
from ggplot import *
ggplot(salary_pie_social_df, aes(x="PIE", y="SALARY_MILLIONS", color="cluster")) +\
geom_point(size=400) + scale_color_gradient(low = 'red', high = 'blue')
distortions = []
for i in range(1, 11):
    km = KMeans(n_clusters=i,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
    km.fit(scaler.transform(numerical_df))
    distortions.append(km.inertia_)
    
plt.plot(range(1,11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.title("Team Valuation Elbow Method Cluster Analysis")
plt.show()
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=0)
y_km = km.fit_predict(scaler.transform(numerical_df))
import numpy as np
from matplotlib import cm
from sklearn.metrics import silhouette_samples
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(scaler.transform(numerical_df),
                                     y_km,
                                     metric='euclidean')
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i)/n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='none',color=color)
    yticks.append((y_ax_lower + y_ax_upper)/2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg,
            color="red",
            linestyle="--")
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.title('Silhouette Plot Team Valuation')
plt.figure(figsize=(20,10))
plt.show()
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
km = KMeans(n_clusters=2,
            random_state=0)
X = scaler.transform(numerical_df)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km==0,0],
            X[y_km==0,1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='cluster 1')
ax1.scatter(X[y_km==1,0],
            X[y_km==1,1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='cluster 2')
ax1.set_title('NBA Team K-means clustering')

from sklearn.cluster import AgglomerativeClustering

X = scaler.transform(numerical_df)
ac = AgglomerativeClustering(n_clusters=2,
                             affinity='euclidean',
                             linkage='complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac==0,0],
             X[y_ac==0,1],
             c='lightblue',
             edgecolor='black',
             marker='o',
            s=40,
            label='cluster 1')
ax2.scatter(X[y_ac==1,0],
            X[y_ac==1,1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='cluster 2')
ax2.set_title('NBA Team Agglomerative clustering')
plt.legend()
plt.show()
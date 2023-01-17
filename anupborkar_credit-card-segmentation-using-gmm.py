%reset -f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans # Class to develop kmeans model
from sklearn import metrics
from sklearn.metrics import silhouette_score # base for clustering
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.mixture import GaussianMixture 

# Use white grid plot background from seaborn
sns.set(font_scale=0.5, style="ticks")
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6})
import warnings
import os
%matplotlib inline
plt.rcParams.update({'figure.max_open_warning': 0}) #just to suppress warning for max plots of 20
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)
#os.chdir("../input/ccdata/")
#os.listdir()            # List all files in the folder
# Display output not only of last command but all commands in a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# Set pandas options to display results
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000
# Load dataset
cc = pd.read_csv("/kaggle/input/ccdata/CC GENERAL.csv")
#Let's try to analyze the dataset based on what is availiable with us
cc.info()
cc.head()
cc.describe()
cc.columns = [i.lower() for i in cc.columns]
cc.columns
del cc['cust_id']
cc.columns
def missing_columns_data(data):
    miss      = data.isnull().sum()
    miss_pct  = 100 * data.isnull().sum()/len(data)
    
    miss_pct      = pd.concat([miss,miss_pct], axis=1)
    missing_cols = miss_pct.rename(columns = {0:'Missings', 1: 'Missing pct'})
    missing_cols = missing_cols[missing_cols.iloc[:,1]!=0].sort_values('Missing pct', ascending = False).round(1)
    
    return missing_cols  

missing = missing_columns_data(cc)
missing
null_counts = cc.isnull().sum()/len(cc);
plt.figure(figsize=(15,4));
plt.xticks(np.arange(len(null_counts))+0.5,null_counts.index,rotation='vertical');
plt.ylabel('Fraction of Rows with missing data');
plt.bar(np.arange(len(null_counts)),null_counts);
# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True);
sns.despine(left=True);

# Plot a kernel density estimate and rug plot
sns.distplot(cc.minimum_payments, hist=True, rug=True, color="r", ax=axes[0, 1]);
sns.distplot(cc.credit_limit, hist=True, rug=True, color="b", ax=axes[0, 0]);

# Plot a filled kernel density estimate
sns.distplot(cc.minimum_payments,  color="g",ax=axes[1, 1]);
sns.distplot(cc.credit_limit, color="m", ax=axes[1, 0]);

plt.setp(axes, yticks=[]);
plt.tight_layout();
values = { 'minimum_payments' : cc['minimum_payments'].median(),
           'credit_limit' : cc['credit_limit'].median()
          }
cc.fillna(value = values, inplace = True)
missing = missing_columns_data(cc)
missing
# Get column names first
names = cc.columns
# Create the Scaler object
scaler = preprocessing.StandardScaler()
# Fit data on the scaler object
scaled_df = scaler.fit_transform(cc)
# Converting the numpy array into a pandas DataFrame
scaled_df = pd.DataFrame(scaled_df, columns=names)
# Normalizing the Data 
normalized_df = normalize(scaled_df) #out
# Converting the numpy array into a pandas DataFrame 
df_out = pd.DataFrame(normalized_df,columns=names) # Normalized Dataframe
df_out.head()
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(18, 18));
for i in range(4):
    for j in range(4):
        sns.distplot(df_out[df_out.columns[4 * i + j]], ax=axs[i,j]);
        sns.despine();
plt.show();
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4,4));
sns.distplot(df_out.tenure, ax=axs);
sns.despine();
plt.show();
sns.set_style("darkgrid");
plt.figure(figsize=(17,10));
sns.boxplot(data=df_out);
#sns.violinplot(data=df_out);
#sns.stripplot(data=df_out);
plt.xticks(rotation=90);
#Using Pearson Correlation
plt.figure(figsize=(12,10));
cor = df_out.corr();
sns.heatmap(cor, annot=True, cmap=plt.cm.PiYG);
plt.title('Correlation Matrix')
plt.show();
# How many clusters?
#     Use either AIC or BIC as criterion
#     Ref: https://en.wikipedia.org/wiki/Akaike_information_criterion
#          https://en.wikipedia.org/wiki/Bayesian_information_criterion
#          https://www.quora.com/What-is-an-intuitive-explanation-of-the-Akaike-information-criterion
bic = []
aic = []
for i in range(8):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100);
    gm.fit(df_out);
    bic.append(gm.bic(df_out));
    aic.append(gm.aic(df_out));
sns.set_style("whitegrid")
# Use white grid plot background from seaborn
sns.set(font_scale=0.5, style="ticks")
sns.set_context("poster", font_scale = .5, rc={"grid.linewidth": 0.6})
#Draw aic ,bic on plot to understand

n_clusters=np.arange(0, 8);
fig, ax = plt.subplots(1, 2, figsize=(12,5));

plt.subplot(1, 2, 1);
plt.plot(n_clusters, aic,marker="o", label='AIC');
plt.title("AIC Scores");
plt.xticks(n_clusters);
plt.xlabel("No. of clusters");
plt.ylabel("Scores");
plt.legend();

plt.subplot(1, 2, 2);
plt.plot(n_clusters, bic, marker="o",label='BIC');
plt.title("BIC Scores");
plt.xticks(n_clusters);
plt.xlabel("No. of clusters");
plt.ylabel("Scores");
plt.legend();

plt.show();
#Gussian Mixture
gm = GaussianMixture(n_components = 3,
                     n_init = 10,
                     max_iter = 100);
gm.fit(df_out);
# Where are the cluster centers
gm.means_
# Did algorithm converge?
gm.converged_
#  How many iterations did it perform?
gm.n_iter_
#  Clusters labels
#gm.predict(df_out)

# Weights of respective gaussians.GMM can also be used as a generator
#to generate data having similar pattern. All three Gaussians may generate data
#as per their specifc pdf (prob density functions). But to generate as per same
#pattern, as the original data, selection of data-sources (the three Gaussian)
#has prob distribution. Weights describe the prob distribution.
#Values of these weights are close to frequency of data-points per cluster
gm.weights_

#  What is the frequency of data-points for the three clusters.
np.unique(gm.predict(df_out), return_counts = True)[1]/len(df_out)
#Draw scatter plot to visualize the GMM output
fig = plt.figure();

plt.scatter(df_out.iloc[:, 0], df_out.iloc[:, 1],
            c=gm.predict(df_out),
            s=5,cmap =plt.cm.Greens
           );
plt.scatter(gm.means_[:, 0], gm.means_[:, 1],
            marker='>',
            s=100,               # marker size
            linewidths=5,      # linewidth of marker edges
            cmap =plt.cm.Greens
            );
plt.show()
gm = GaussianMixture(
                     n_components = 3,
                     n_init = 10,
                     max_iter = 100)
gm.fit(df_out)

tsne = TSNE(n_components = 2, perplexity=50)
tsne_out = tsne.fit_transform(df_out)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=5,              # marker size
            linewidths=5,      # linewidth of marker edges
            c=gm.predict(df_out)   # Colour as per gmm
            )
#df_out['cluster'] = gm.predict(df_out)
#len(df_out.cluster.unique())
arr=gm.predict(df_out);
df_cluster=pd.DataFrame(arr,columns=['cluster'])
df_out_cluster = pd.concat([df_out,df_cluster],axis=1)
#df_cluster.head()
df_out_cluster.head()
# Number of clients by cluster
df_out_cluster['cluster'].value_counts().plot.bar(figsize=(10,5), title='Clusterwise No. of Customers');

df_out_cluster['cluster'].value_counts()
#pairwise relationships of key_features

key_features = ["balance", "purchases", "cash_advance","credit_limit", "payments", "minimum_payments", "tenure","cluster"]

#key_features.append("cluster")
plt.figure(figsize=(25,25))
sns.pairplot( df_out_cluster[key_features], hue="cluster")
densities = gm.score_samples(normalized_df)
density_threshold = np.percentile(densities,4)

# anomalies data
anomalies = normalized_df[densities < density_threshold] # Data of anomalous customers
df_anomaly=pd.DataFrame(anomalies,columns=df_out.columns)
# Unanomalous data
unanomalous = normalized_df[densities >= density_threshold] # Data of unanomalous customers
df_unanomaly=pd.DataFrame(unanomalous,columns=df_out.columns)
#Density Plot Function to draw plots for Anomalous and Un-Anamalous data
def densityplots(df1,df2, label1 = "Anomalous",label2 = "Normal"):
    # df1 and df2 are two dataframes
    # As number of features are 17, we have 20 axes
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(18,15))
    ax = axes.flatten()
    fig.tight_layout()
    # Do not display 18th, 19th and 20th axes
    axes[3,3].set_axis_off()
    axes[3,2].set_axis_off()
    axes[3,4].set_axis_off()
    # Below 'j' is not used.
    for i,j in enumerate(df1.columns):
        # https://seaborn.pydata.org/generated/seaborn.distplot.html
        # For every i, draw two overlapping density plots in different colors
        sns.distplot(df1.iloc[:,i],
                     ax = ax[i],
                     kde_kws={"color": "k", "lw": 3, "label": label1},   # Density plot features
                     hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"}) # Histogram features
        sns.distplot(df2.iloc[:,i],
                     ax = ax[i],
                     kde_kws={"color": "red", "lw": 3, "label": label2},
                     hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "b"})

# Draw density plots now from two dataframes of anomalous dataframe (df_anomaly) and un-anomalous dataframe (df_unanomaly)
densityplots(df_anomaly, df_unanomaly, label1 = "Anomalous",label2 = "Normal")
df_anomaly['type'] = 'anomalous'  
df_unanomaly['type'] = 'unanomalous'
df_all = pd.concat([df_anomaly,df_unanomaly])
fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(18, 18));
for i in range(4):
    for j in range(4):
        sns.boxplot(x = df_all['type'],  y = df_all[df_all.columns[4 * i + j]], ax=axs[i,j]);
        sns.despine();
plt.show();
fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4,4));
sns.boxplot(x = df_all['type'],y=df_all.tenure, ax=axs);
sns.despine();
plt.show();
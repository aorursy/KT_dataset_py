%reset -f  

import warnings
warnings.filterwarnings("ignore")

# 1.1 Data manipulation library
import pandas as pd
import numpy as np
%matplotlib inline

# 1.2 OS related package

import os

# 1.3 Modeling librray
# 1.3.1 Scale data

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
# 1.4 Plotting library

import seaborn as sns
import matplotlib.pyplot as plt

# 1.5 Import GaussianMixture class

from sklearn.mixture import GaussianMixture

# 1.6 TSNE
from sklearn.manifold import TSNE

# DateFrame object is created while reading file available at particular location given below

cc=pd.read_csv("../input/ccdata/CC GENERAL.csv")
cc.head()
cc.shape
cc.info()
cc.columns = [i.lower() for i in cc.columns]
cc.columns
cc.drop(columns="cust_id",inplace=True)
cc.columns
cc.isnull().sum()
sns.distplot(cc.credit_limit,color="b")
sns.distplot(cc.minimum_payments,color="g")
values = {'minimum_payments' :   cc['minimum_payments'].median(),
          'credit_limit'     :   cc['credit_limit'].median()
         }

cc.fillna(value=values,inplace=True)
cc.isnull().sum()
ss =  StandardScaler()
out = ss.fit_transform(cc)
out = normalize(out)
col_names=cc.columns
df_out=pd.DataFrame(out,columns=col_names)
df_out.head()
fig = plt.figure(figsize=(20,20))

for i in range(17):
    plt.subplot(6,3,i+1)
    sns.distplot(df_out[df_out.columns[i]])
    
fig = plt.figure(figsize=(15, 10))

sns.boxplot(data=df_out)

plt.xticks(rotation=90)
sns.jointplot(x="balance", y="credit_limit", data=df_out,color="g")
sns.jointplot(x="balance", y="credit_limit", data=df_out,kind="kde",color="g")
fig = plt.figure(figsize=(20, 10))

heatmap = sns.heatmap(df_out.corr(),annot = True)

heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':15})
bic = []
aic = []
for i in range(3):
    gm = GaussianMixture(
                     n_components = i+1,
                     n_init = 10,
                     max_iter = 100)
    gm.fit(df_out)
    bic.append(gm.bic(df_out))
    aic.append(gm.aic(df_out))
fig = plt.figure()
plt.plot([1,2,3], aic,marker="o",label="aic",color="b")
plt.plot([1,2,3], bic,marker="o",label="bic",color="r")
plt.legend()
plt.show()
tsne = TSNE(n_components = 2)
tsne_out = tsne.fit_transform(df_out)
plt.scatter(tsne_out[:, 0], tsne_out[:, 1],
            marker='x',
            s=20,                   # marker size
            linewidths=5,           # linewidth of marker edges
            c=gm.predict(df_out)    # Colour as per gmm
            )
# Anomalous points are those that are in low-density region Or where density is in low-percentile of 4%


densities = gm.score_samples(df_out)              #score_samples() method gives score or density of a point at any location.
densities

density_threshold = np.percentile(densities,4)
density_threshold

anomalies = df_out[densities < density_threshold]
anomalies
anomalies.shape               
unanomalies = df_out[densities >= density_threshold]
unanomalies
unanomalies.shape    
df_anomaly = pd.DataFrame(anomalies, columns = df_out.columns)

df_unanomaly = pd.DataFrame(unanomalies, columns =df_out.columns)
def densityplots(df1,df2, label1 = "Anomalous",label2 = "Normal"):
    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15,15))
    ax = axes.flatten()
    fig.tight_layout()
    # Do not display 18th, 19th and 20th axes
    axes[3,3].set_axis_off()
    axes[3,2].set_axis_off()
    axes[3,4].set_axis_off()
    # Below 'j' is not used.
    for i,j in enumerate(df1.columns):
        sns.distplot(df1.iloc[:,i],
                     ax = ax[i],
                     kde_kws={"color": "k", "lw": 3, "label": label1},   # Density plot features
                     hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "g"}) # Histogram features
        sns.distplot(df2.iloc[:,i],
                     ax = ax[i],
                     kde_kws={"color": "red", "lw": 3, "label": label2},
                     hist_kws={"histtype": "step", "linewidth": 3,"alpha": 1, "color": "b"})
densityplots(df_anomaly, df_unanomaly, label2 = "Unanomalous")
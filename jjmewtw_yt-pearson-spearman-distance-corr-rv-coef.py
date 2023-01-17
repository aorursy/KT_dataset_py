!pip install hoggorm #not very popular, let's install



import numpy as np

import pandas as pd

import matplotlib as plt

import seaborn as sns

import matplotlib.pyplot

import scipy.stats as scp

from scipy.spatial import distance

from scipy.spatial.distance import pdist, squareform

import hoggorm as hogg

import sklearn.preprocessing as skl



Variables = ['video_id', 'views', 'likes', 'dislikes', 'comment_count']

        

us_yt = pd.read_csv('../input/youtube-new/USvideos.csv', usecols=Variables) #USA

ca_yt = pd.read_csv('../input/youtube-new/CAvideos.csv', usecols=Variables) #Canada

de_yt = pd.read_csv('../input/youtube-new/DEvideos.csv', usecols=Variables) #Germany

fr_yt = pd.read_csv('../input/youtube-new/FRvideos.csv', usecols=Variables) #France

gb_yt = pd.read_csv('../input/youtube-new/GBvideos.csv', usecols=Variables) #Great Brittain

in_yt = pd.read_csv('../input/youtube-new/INvideos.csv', usecols=Variables) #India

jp_yt = pd.read_csv('../input/youtube-new/JPvideos.csv', usecols=Variables) #Japan

kr_yt = pd.read_csv('../input/youtube-new/KRvideos.csv', usecols=Variables) #South Korea

mx_yt = pd.read_csv('../input/youtube-new/MXvideos.csv', usecols=Variables) #Mexico

ru_yt = pd.read_csv('../input/youtube-new/RUvideos.csv', usecols=Variables) #Russia
N = (us_yt.dtypes == 'int64')

Numeric = list(N[N].index)
# Center the data before computation of RV coefficients

us_yt_sc = skl.scale(us_yt[Numeric], axis=0, with_mean=True)

ca_yt_sc = skl.scale(ca_yt[Numeric], axis=0, with_mean=True)

de_yt_sc = skl.scale(de_yt[Numeric], axis=0, with_mean=True)

fr_yt_sc = skl.scale(fr_yt[Numeric], axis=0, with_mean=True)

gb_yt_sc = skl.scale(gb_yt[Numeric], axis=0, with_mean=True)

in_yt_sc = skl.scale(in_yt[Numeric], axis=0, with_mean=True)

jp_yt_sc = skl.scale(jp_yt[Numeric], axis=0, with_mean=True)

kr_yt_sc = skl.scale(kr_yt[Numeric], axis=0, with_mean=True)

mx_yt_sc = skl.scale(mx_yt[Numeric], axis=0, with_mean=True)

ru_yt_sc = skl.scale(ru_yt[Numeric], axis=0, with_mean=True)
us_yt['Country'] = "US"

ca_yt['Country'] = "CA"

de_yt['Country'] = "DE"

fr_yt['Country'] = "FR"

gb_yt['Country'] = "GB"

in_yt['Country'] = "IN"

jp_yt['Country'] = "JP"

kr_yt['Country'] = "KR"

mx_yt['Country'] = "MX"

ru_yt['Country'] = "RU"

 

df = pd.concat([us_yt, ca_yt, de_yt,fr_yt,gb_yt,in_yt,jp_yt,kr_yt,mx_yt,ru_yt] )

df.reset_index

df.head()
Missing_Percentage = (df.isnull().sum()).sum()/np.product(df.shape)*100

print("The number of missing entries: " + str(round(Missing_Percentage,8)) + " %")
PearsonCorr = df.corr(method="pearson")

matplotlib.pyplot.figure(figsize=(10,10))

sns.heatmap(PearsonCorr, vmax=.9, square=True)
SpearmanCorr = df.corr(method="spearman")

matplotlib.pyplot.figure(figsize=(10,10))

sns.heatmap(SpearmanCorr, vmax=.9, square=True)
DistanceCorr = pd.DataFrame([[0.00]*4,[0.00]*4,[0.00]*4,[0.00]*4], columns=['views','likes','dislikes','comment_count'], index=['views','likes','dislikes','comment_count'])



for i in range(0,4):

    DistanceCorr.views[i] = -distance.correlation(df['views'],df[DistanceCorr.iloc[:, [i]].columns]) + 1 

    DistanceCorr.likes[i] = -distance.correlation(df['likes'],df[DistanceCorr.iloc[:, [i]].columns]) + 1

    DistanceCorr.dislikes[i] = -distance.correlation(df['dislikes'],df[DistanceCorr.iloc[:, [i]].columns]) + 1

    DistanceCorr.comment_count[i] = -distance.correlation(df['comment_count'],df[DistanceCorr.iloc[:, [i]].columns]) + 1



matplotlib.pyplot.figure(figsize=(10,10))

sns.heatmap(DistanceCorr, vmax=.9, square=True)
PearsonCorr
SpearmanCorr
DistanceCorr
def distcorr(X, Y):

    # Compute the distance correlation function

    X = np.atleast_1d(X)

    Y = np.atleast_1d(Y)

    if np.prod(X.shape) == len(X):

        X = X[:, None]

    if np.prod(Y.shape) == len(Y):

        Y = Y[:, None]

    X = np.atleast_2d(X)

    Y = np.atleast_2d(Y)

    n = X.shape[0]

    if Y.shape[0] != X.shape[0]:

        raise ValueError('Number of samples must match')

    a = squareform(pdist(X))

    b = squareform(pdist(Y))

    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()

    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()

    

    dcov2_xy = (A * B).sum()/float(n * n)

    dcov2_xx = (A * A).sum()/float(n * n)

    dcov2_yy = (B * B).sum()/float(n * n)

    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

    return dcor
# Data sets have to have same length for this method

Shortest = min(ca_yt_sc.shape[0],de_yt_sc.shape[0],fr_yt_sc.shape[0],gb_yt_sc.shape[0],in_yt_sc.shape[0],

                jp_yt_sc.shape[0],kr_yt_sc.shape[0],mx_yt_sc.shape[0],ru_yt_sc.shape[0],us_yt_sc.shape[0])



Upper = 1000



RV = pd.DataFrame(hogg.RVcoeff([ ca_yt_sc[0:Upper] , de_yt_sc[0:Upper], fr_yt_sc[0:Upper], gb_yt_sc[0:Upper], in_yt_sc[0:Upper], 

                    jp_yt_sc[0:Upper] , kr_yt_sc[0:Upper], mx_yt_sc[0:Upper], ru_yt_sc[0:Upper], us_yt_sc[0:Upper] ]),

                 columns=['CA','DE','FR','GB','IN','JP','KR','MX','RU','US'])
matplotlib.pyplot.figure(figsize=(10,10))

sns.heatmap(RV, vmax=.9, square=True)
Check =  any(item in ca_yt.video_id for item in us_yt.video_id)

Check
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE 

from sklearn.decomposition import PCA

import umap

%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart.csv')

df.head(5)
#Checking missing values

df.isnull().sum()
feat=df.drop(['target'],axis=1)
target=df['target']
X=df.drop(['target'],axis=1)

X.corrwith(df['target']).plot.bar(

        figsize = (20, 10), title = "Correlation with Target", fontsize = 20,

        rot = 90, grid = True)
from sklearn.decomposition import PCA

pca = PCA(n_components=7)

pca_result = pca.fit_transform(feat.values)
plt.plot(range(7), pca.explained_variance_ratio_)

plt.plot(range(7), np.cumsum(pca.explained_variance_ratio_))

plt.title("Component-wise and Cumulative Explained Variance")
def pcaR (OOOOO0OOO0OOO00O0 ,O00OO000000O00O00 ):#line:1

	OO00OOOO0O0OOO0O0 =OO00OOOO0O0OOO0O0 =['Dimension {}'.format (OO0O00O00OO0O0OO0 )for OO0O00O00OO0O0OO0 in range (1 ,len (O00OO000000O00O00 .components_ )+1 )]#line:4

	OO0O00OOOOO0O0OO0 =pd .DataFrame (np .round (O00OO000000O00O00 .components_ ,4 ),columns =list (OOOOO0OOO0OOO00O0 .keys ()))#line:7

	OO0O00OOOOO0O0OO0 .index =OO00OOOO0O0OOO0O0 #line:8

	O00O0OOO0O00O0O00 =O00OO000000O00O00 .explained_variance_ratio_ .reshape (len (O00OO000000O00O00 .components_ ),1 )#line:11

	O000O0OOO0OO00OO0 =pd .DataFrame (np .round (O00O0OOO0O00O0O00 ,4 ),columns =['Explained Variance'])#line:12

	O000O0OOO0OO00OO0 .index =OO00OOOO0O0OOO0O0 #line:13

	O0O0000OOOOO00OOO ,OOOOO00OO0O00000O =plt .subplots (figsize =(14 ,8 ))#line:16

	OO0O00OOOOO0O0OO0 .plot (ax =OOOOO00OO0O00000O ,kind ='bar');#line:19

	OOOOO00OO0O00000O .set_ylabel ("Feature Weights")#line:20

	OOOOO00OO0O00000O .set_xticklabels (OO00OOOO0O0OOO0O0 ,rotation =0 )#line:21

	for OO0000O00O00000OO ,OO0O0O0O000OOO000 in enumerate (O00OO000000O00O00 .explained_variance_ratio_ ):#line:25

		OOOOO00OO0O00000O .text (OO0000O00O00000OO -0.40 ,OOOOO00OO0O00000O .get_ylim ()[1 ]+0.05 ,"Explained Variance\n          %.4f"%(OO0O0O0O000OOO000 ))#line:26

	return pd .concat ([O000O0OOO0OO00OO0 ,OO0O00OOOOO0O0OO0 ],axis =1 )#line:29

pca_results =pcaR (feat ,pca )
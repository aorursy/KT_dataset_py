# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns# data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("../input/home-credit-default-risk/application_train.csv")
train.head()
missing_values = train.isna().sum()/len(train)*100
missing_values[missing_values>0].sort_values(ascending  = False)
cols = train.columns
a = train.isna().sum()/len(train)*100
variable = []
for i in range(0,len(cols)):
    if a[i]>=60:
        variable.append(cols[i])
print(variable)        
for col in cols:
    train[col].fillna(train[col].mode()[0],inplace = True)
missing_values = train.isna().sum()/len(train)*100
missing_values
train.var().sort_values(ascending = False)
numeric = train.select_dtypes(include=[np.number])
var = numeric.var()
variance = []
for i in range(len(var)):
    if var[i]>=30:
        variance.append(var[i])
variance
df=train.drop('TARGET', 1)
f , ax = plt.subplots(figsize = (14,12))
plt.title('Correlation of Features- HeatMap',y=1,size=16)
sns.heatmap(df.corr(),square = True,  vmax=0.8,annot = False)
from sklearn.ensemble import RandomForestRegressor
df=df.drop(['SK_ID_CURR','DAYS_ID_PUBLISH'], axis=1)
model = RandomForestRegressor(random_state=1, max_depth=10)
df=pd.get_dummies(df)
model.fit(df,train.TARGET)
features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances[0:20])  # top 20 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))
# Construct our Linear Regression model
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
lr = LinearRegression(normalize=True)
lr.fit(df, train.TARGET)
#stop the search when only the last feature is left
rfe = RFE(lr, n_features_to_select=1, verbose =3 )
rfe.fit(df, train.TARGET)

from sklearn.preprocessing import MinMaxScaler
ranks["RFE"] = ranking(list(map(float, rfe.ranking_)), train.columns, order=-1)
# Create empty dictionary to store the mean value calculated from all the scores
r = {}
for name in train.columns:
    r[name] = round(np.mean([ranks[method][name] 
                             for method in ranks.keys()]), 2)
# Put the mean scores into a Pandas dataframe
meanplot = pd.DataFrame(list(r.items()), columns= ['Feature','Mean Ranking'])
# Sort the dataframe
meanplot = meanplot.sort_values('Mean Ranking', ascending=False)
# Let's plot the ranking of the features
sns.factorplot(x="Mean Ranking", y="Feature", data = meanplot, kind="bar",size=16, aspect=0.75, palette='coolwarm')
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
train.head()
train_data = np.array(train,dtype = 'float32')
img = []
for i in range(len(train)):
    image = train_data[i].flatten()
    img.append(image)
img = np.array(img,dtype = 'float32')    
image.shape

train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv",sep=',')    # Give the complete path of your train.csv file
feat_cols = [ 'pixel'+str(i) for i in range(img.shape[1]) ]
df = pd.DataFrame(img,columns=feat_cols)
df['label'] = train['label']
df.head()
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components = 3).fit_transform(df[feat_cols].values)
%matplotlib inline
import matplotlib.pyplot as plt
plt.figure(figsize=(16,10))
plt.title('Factor Analysis Components')
plt.scatter(fa[:,0], fa[:,1],c='r',s=10)
plt.scatter(fa[:,1], fa[:,2],c='b',s=10)
plt.scatter(fa[:,2],fa[:,0],c='g',s=10)
plt.legend(("First Factor","Second Factor","Third Factor"))
from sklearn.decomposition import PCA
pca = PCA(n_components=4)
pca_result = pca.fit_transform(df[feat_cols].values)
plt.figure(figsize=(14,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance');
index = np.arange(len(pca.explained_variance_ratio_))
plt.figure(figsize=(14,6))
plt.title('Principal Component Analysis')
plt.bar(index, pca.explained_variance_ratio_*100)
plt.xlabel('Principal Component', fontsize=10)
plt.ylabel('Explained Variance', fontsize=10)
plt.xticks(index, pca.explained_variance_ratio_*100, fontsize=10, rotation=30)
plt.show()
plt.figure(figsize=(16,10))
plt.plot(range(4), pca.explained_variance_ratio_)
plt.plot(range(4), np.cumsum(pca.explained_variance_ratio_))
plt.title("PCA - Cumulative Explained Variance vs. Component-Explained Variance ")
plt.legend(("Component - Explained Variance","Cumulative Sum - Explained Variance"))
plt.scatter(pca_result[:, 0], pca_result[:, 1],pca_result[:, 2], pca_result[:, 3],
            edgecolor='none', alpha=0.9,
            cmap=plt.cm.get_cmap('Spectral', 8))
plt.colorbar();
from sklearn.decomposition import TruncatedSVD 
svd = TruncatedSVD(n_components=3, random_state=42).fit_transform(df[feat_cols].values)
svd.shape
plt.figure(figsize=(16,10))
plt.title('SVD Components')
plt.scatter(svd[:,0], svd[:,1],c='r',s=10)
plt.scatter(svd[:,1], svd[:,2],c='b',s=10)
plt.scatter(svd[:,2],svd[:,0],c='g',s=10)
plt.legend(("Principal Component 1","Principal Component 2","Principal Component 3"))
from sklearn.decomposition import FastICA 
ICA = FastICA(n_components=3, random_state=12) 
X=ICA.fit_transform(df[feat_cols].values)
plt.figure(figsize=(16,8))
plt.title('ICA Components')
plt.scatter(X[:,0], X[:,1],c='r',s=10)
plt.scatter(X[:,1], X[:,2],c='b',s=10)
plt.scatter(X[:,2], X[:,0],c='g',s=10)
plt.legend(("ICA Component 1","ICA Component 2","ICA Component 3"))

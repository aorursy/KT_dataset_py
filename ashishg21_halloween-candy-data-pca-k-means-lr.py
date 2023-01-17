import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

from IPython.display import display
warnings.filterwarnings('ignore') # ignore warnings.

%config IPCompleter.greedy = True # autocomplete feature.

pd.options.display.max_rows = None # set maximum rows that can be displayed in notebook.

pd.options.display.max_columns = None # set maximum columns that can be displayed in notebook.

pd.options.display.precision = 2 # set the precision of floating point numbers.
# # Check the encoding of data. Use ctrl+/ to comment/un-comment.



# import chardet



# rawdata = open('candy-data.csv', 'rb').read()

# result = chardet.detect(rawdata)

# charenc = result['encoding']

# print(charenc)

# print(result) # It's utf-8 with 99% confidence.
df = pd.read_csv('../input/candy-data.csv', encoding='utf-8')

df.drop_duplicates(inplace=True) # drop duplicates if any.

df.shape # num rows x num columns.
(df.isnull().sum()/len(df)*100).sort_values(ascending=False)
df.head()
df['winpercent'] = df['winpercent']/100
df['sugarbyprice'] = df['sugarpercent'].div(df['pricepercent']) # higher value means the candy is sweet as well as cheap.

df['winbyprice'] = df['winpercent'].div(df['pricepercent']) # higher value means the candy is more liked as well as cheap.
categorival_vars = ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar',

                    'pluribus']

numerical_vars = ['sugarpercent', 'pricepercent', 'winpercent', 'sugarbyprice', 'winbyprice']
df['competitorname'] = df['competitorname'].str.replace('Õ', "'") # Special character was appearing in name of candy.

df.sort_values(by=['winpercent', 'sugarpercent'], ascending=False).head(10)
df[df['chocolate']==0].sort_values(by=['winpercent', 'sugarpercent'], ascending=False).head(10)
df.sort_values(by=['winbyprice', 'winpercent'], ascending=False).head(10)
df.sort_values(by=['sugarpercent', 'winpercent'], ascending=False).head(10)
df[(df['chocolate']==1)&(df['fruity']==1)]
plt.figure(figsize = (20,8))        

sns.heatmap(df.corr(),annot=True, cmap = 'coolwarm')
# Improting the PCA module. 



from sklearn.decomposition import PCA # import.

pca = PCA(svd_solver='randomized', random_state=123) #instantiate.

pca.fit(df.drop('competitorname', axis=1)) # fit.
# Making the screeplot - plotting the cumulative variance against the number of components



fig = plt.figure(figsize = (20,5))

ax = plt.subplot(121)

plt.plot(pca.explained_variance_ratio_)

plt.xlabel('principal components')

plt.ylabel('explained variance')



ax2 = plt.subplot(122)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')



plt.show()
# what percentage of variance in data can be explained by first 2,3 and 4 principal components respectively?

(pca.explained_variance_ratio_[0:2].sum().round(3),

pca.explained_variance_ratio_[0:3].sum().round(3),

pca.explained_variance_ratio_[0:4].sum().round(3))
# we'll use first 2 principal components as it retains 95% of variance.



df_pca_2_comp = pd.DataFrame({'PC1':pca.components_[0],'PC2':pca.components_[1], 'Feature':df.drop(

                              'competitorname', axis=1).columns})

# df_pca_2_comp
# we can visualize what the principal components seem to capture.



fig = plt.figure(figsize = (6,6))

plt.scatter(df_pca_2_comp.PC1, df_pca_2_comp.PC2)

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

for i, txt in enumerate(df_pca_2_comp.Feature):

    plt.annotate(txt, (df_pca_2_comp.PC1[i],df_pca_2_comp.PC2[i]))

plt.tight_layout()

plt.show()
df_pca = pca.transform(df.drop('competitorname', axis=1)) # our data transformed with new features as principal components.

df_pca = df_pca[:, 0:2] # Since we require first two principal components only.
from sklearn.preprocessing import StandardScaler



standard_scaler = StandardScaler()

df_s = standard_scaler.fit_transform(df_pca) # s in df_s stands for scaled.
sns.pairplot(pd.DataFrame(df_s)) # Try to get some intuiton of data.
from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H
hopkins(pd.DataFrame(df_s))
from sklearn.cluster import KMeans # import.



# silhouette scores to choose number of clusters.

from sklearn.metrics import silhouette_score

def sil_score(df):

    sse_ = []

    for k in range(2, 15):

        kmeans = KMeans(n_clusters=k, random_state=123).fit(df_s) # fit.

        sse_.append([k, silhouette_score(df, kmeans.labels_)])

    plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])



sil_score(df_s)
# sum of squared distances.



def plot_ssd(df):

    ssd = []

    for num_clusters in list(range(1,19)):

        model_clus = KMeans(n_clusters = num_clusters, max_iter=50, random_state=123)

        model_clus.fit(df)

        ssd.append(model_clus.inertia_)

    plt.plot(ssd)



plot_ssd(df_s)
# K-means with K=2.

km2c = KMeans(n_clusters=2, max_iter=50, random_state=93)

km2c.fit(df_s)
# creation of data frame with original features for analysis of clusters formed.



df_dummy = pd.DataFrame.copy(df)

dfkm2c = pd.concat([df_dummy, pd.Series(km2c.labels_)], axis=1)

dfkm2c.rename(columns={0:'Cluster ID'}, inplace=True)

# dfkm2c.head()
# creation of data frame with features as principal components for analysis of clusters formed.



df_dummy = pd.DataFrame.copy(pd.DataFrame(df_s))

dfpcakm2c = pd.concat([df_dummy, pd.Series(km2c.labels_)], axis=1)

dfpcakm2c.columns = ['PC1', 'PC2', 'Cluster ID']
sns.pairplot(data=dfpcakm2c, vars=['PC1', 'PC2'], hue='Cluster ID')
# K-means with K=5.

km5c = KMeans(n_clusters=5, max_iter=50, random_state=123)

km5c.fit(df_s)
# creation of data frame with original features for analysis of clusters formed.



df_dummy = pd.DataFrame.copy(df)

dfkm5c = pd.concat([df_dummy, pd.Series(km5c.labels_)], axis=1) # df-dataframe, km-kmeans, 5c-5clusters.

dfkm5c.rename(columns={0:'Cluster ID'}, inplace=True)

# dfkm5c.head()
# creation of data frame with features as principal components for analysis of clusters formed.



df_dummy = pd.DataFrame.copy(pd.DataFrame(df_s))

dfpcakm5c = pd.concat([df_dummy, pd.Series(km5c.labels_)], axis=1)

dfpcakm5c.columns = ['PC1', 'PC2', 'Cluster ID']
sns.pairplot(data = dfpcakm5c, vars=['PC1', 'PC2'], hue='Cluster ID')
dfkm5c.groupby('Cluster ID').mean()
dfkm5c[dfkm5c['Cluster ID']!=0]
dfkm5c['Cluster ID'] = dfkm5c['Cluster ID'].map(lambda x: 1 if (x!=0) else 0)
dfkm5c.groupby('Cluster ID').mean()
X = df.drop(['competitorname', 'winpercent', 'sugarpercent', 'pricepercent', 'sugarbyprice', 'winbyprice'], axis=1)

y = df['winpercent']



from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()
from sklearn import linear_model # import.

lr_rdg = linear_model.Ridge(random_state=123) # instantiate.



# Perform cross-validation.

from sklearn.model_selection import GridSearchCV

hyperparameters = {'alpha': [0.01, 0.1, 1, 10, 100, 1000]}

model_cv = GridSearchCV(estimator = lr_rdg, param_grid = hyperparameters, cv=10, scoring= 'neg_mean_absolute_error')

#lr_rdg.get_params().keys() # hyperparameters that we can set.



model_cv.fit(X, y) # fit.
cv_results = pd.DataFrame(model_cv.cv_results_)

# cv_results.head()



# Plotting mean test and train scoes with alpha.

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')



# Plotting.

plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])

plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])

plt.xlabel('alpha')

plt.ylabel('Negative Mean Absolute Error')



plt.title("Negative Mean Absolute Error and alpha")

plt.legend(['train score', 'test score'], loc='upper left')

plt.show()
model_cv.best_params_
alpha = 1

ridge = linear_model.Ridge(alpha=alpha)

ridge.fit(X, y)
ridge.intercept_ # constant term.
for x,y in zip(X.columns, ridge.coef_): # coefficients of features.

    print(x, y*100)
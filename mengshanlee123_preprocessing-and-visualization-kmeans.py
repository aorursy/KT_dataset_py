import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/ccdata/CC GENERAL.csv')

data_raw=data.copy()
data.dtypes
pd.isnull(data).sum()
missing_index=data[data['CREDIT_LIMIT'].isnull()].index.to_list()

data=data.drop(index=missing_index[0])
sns.kdeplot(data['MINIMUM_PAYMENTS'], shade=True) 

plt.title('Kernel Density Estimation Plot') 

data['MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median())
pd.isnull(data).sum()
data_dist=data.iloc[:,1:17]

data_columns=data_dist.columns



r,c=0,0

fig, axes=plt.subplots(4,4, figsize=(20,16))

#plt.tight_layout()

for i in data_columns:

    sns.distplot(data[i], ax=axes[r,c])

    c += 1

    if c == 4: 

        r += 1

        c=0

    if r == 4: break

plt.suptitle('Kernel Density Estimation Plot', fontsize=15)
from scipy import stats

data1=data.drop(columns=['CUST_ID', 'TENURE']) # drop string feature and features with meaningful range

z_score=pd.DataFrame(np.abs(stats.zscore(data1)), columns=data1.columns) # calculate z-score



# Find out features with more than 2% outliers (absolute z-score >3)

z_score3=[]

over3_index=[] 

for i in z_score.columns:

    indexs=z_score.index[z_score[i] > 3].tolist()

    ans=i, "{:.3f}".format(len(indexs)/len(z_score)), indexs

    z_score3.append(ans) 

    if len(indexs)/len(z_score) > 0.02:

        over3_index.append(i)  



# remove 'BALANCE' and 'CASH_ADVANCE' since thay are regarded as high discriminative features

del over3_index[0]

del over3_index[1]



# replace 'BALANCE_FREQUENCY','CASH_ADVANCE_FREQUENCY', and 'PURCHASES_TRX' with their square root value

for i in over3_index:

    data1['sqrt_%s' % i]=data1[i].apply(np.sqrt)
print('feature: ', list(data1.columns))

print('data shape: ', data1.shape)
corr_coef=data[1:].corr()



# Heatmap

plt.figure(figsize=(25, 25))

sns.heatmap(corr_coef, cmap='Greens', annot=True, annot_kws={'size':14},

            xticklabels=corr_coef.columns,

            yticklabels=corr_coef.columns)

plt.title('Correlation Matrix')



# Find out feature pairs whose coefficient >= 0.7

corr_cols=corr_coef.columns.to_list() 

signif_corr=[]

for i in range(len(corr_cols)):

    col=corr_cols[i]

    signif_corr.append(abs(corr_coef[col])[abs(corr_coef[col]) >= 0.7])

signif_corr_df=pd.DataFrame(signif_corr)

#signif_corr_df['PURCHASES']['ONEOFF_PURCHASES'] 
sns.kdeplot(data1['INSTALLMENTS_PURCHASES'], shade=True)

sns.kdeplot(data1['ONEOFF_PURCHASES'], shade=True)

sns.kdeplot(data1['PURCHASES'], shade=True)

plt.title('Kernel Density Estimation Plot')
sns.kdeplot(data1['PURCHASES_INSTALLMENTS_FREQUENCY'], shade=True)

sns.kdeplot(data1['ONEOFF_PURCHASES_FREQUENCY'], shade=True)

sns.kdeplot(data1['PURCHASES_FREQUENCY'], shade=True)

plt.title('Kernel Density Estimation Plot')
data1['avg_oneoff_purchases']=data1['ONEOFF_PURCHASES']/data1['ONEOFF_PURCHASES_FREQUENCY']

data1['avg_oneoff_purchases']=data1['avg_oneoff_purchases'].fillna(0)



data1['avg_installment_purchases']=data1['INSTALLMENTS_PURCHASES']/data1['PURCHASES_INSTALLMENTS_FREQUENCY']

data1['avg_installment_purchases']=data1['avg_installment_purchases'].fillna(0)



data1['avg_cash_advance']=data1['CASH_ADVANCE']/data1['CASH_ADVANCE_TRX']

data1['avg_cash_advance']=data1['avg_cash_advance'].fillna(0)
import math

digit_index=list(data1.columns)



for i in digit_index:

    max_v=math.ceil(data1[i].describe()['max'])

    min_v=math.floor(data1[i].describe()['min'])

    bins_range=np.arange(min_v, max_v, (max_v-min_v)/8)    

    data1['digit_%s' % i]=np.digitize(data1[i], bins=bins_range)

    #print(np.unique(data1['digit_%s' % i], return_counts=True))



data1['CUST_ID']=data['CUST_ID']

data1['TENURE']=data['TENURE']
print('data shape: ', data1.shape)
from sklearn.preprocessing import Normalizer, RobustScaler, OneHotEncoder

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names] 
categ=list(data1.columns)[22:44]

categ1=np.delete(categ,[1, 9, 11]).tolist()

numer=['TENURE']



print('Categorical Features: ', categ1)

print('Numerical Feature: ', numer)
numerical_pipeline=Pipeline([('selector', DataFrameSelector(numer)),

                             ('RobustScaler', RobustScaler())])



categorical_pipeline=Pipeline([('selector', DataFrameSelector(categ1)),

                             ('OneHotEncoder', OneHotEncoder())])



selector_pipeline=FeatureUnion([('numerical_pipeline', numerical_pipeline),

                                ('categorical_pipeline', categorical_pipeline)])
def silhouette_score_cal(estimator,data):       

    preprocess=FeatureUnion([('selector_pipeline', selector_pipeline), 

                             ('Normalizer', Normalizer(norm='l2')),

                             ('pca', PCA(n_components=15))])            

    trans_results=preprocess.fit_transform(data)          

    clusters=estimator.fit_predict(data)

    score=silhouette_score(trans_results, clusters)

    return score
categ_copy=categ1.copy()

categ_copy.append('TENURE')

data_model=data1[categ_copy]



preprocess=FeatureUnion([('selector_pipeline', selector_pipeline), 

                             ('Normalizer', Normalizer(norm='l2')),

                             ('pca', PCA(n_components=15))])



trans_results=preprocess.fit_transform(data_model)  # for visualization    

kmeans=Pipeline([('preprocess', preprocess), ('kmeans', KMeans())])     

search_space=[{'kmeans__n_clusters':np.arange(3,10)}] # test various(3-9) n_clusters

cv = [(slice(None), slice(None))]



gs=GridSearchCV(estimator=kmeans,param_grid=search_space, 

                scoring=silhouette_score_cal,cv=cv, n_jobs=-1)



best_model=gs.fit(data_model)
print('best model - number of cluster: ', best_model.best_params_)

print('best model - Silhouette Score: ', best_model.best_score_)

grid_predict=best_model.predict(data_model)

data1['cluster']=grid_predict

grid_results=best_model.cv_results_

print('number of observations in each cluster: ', list(np.unique(grid_predict, return_counts=True)[1])) 
grid_scores=grid_results['mean_test_score']

plt.plot(range(3,10), grid_results['mean_test_score'])

plt.title('Silhouette Score under Various Number of Cluster')
df_visual=pd.DataFrame(TruncatedSVD(n_components=2).fit_transform(trans_results), columns=['p1','p2'])



plt.figure(figsize=(10,8))

plt.scatter(df_visual['p1'], df_visual['p2'], c=grid_predict, cmap=plt.cm.summer)

plt.title('Clustering Results Visualization')
for feature in list(data.columns[1:]):

    g=sns.FacetGrid(data1, col='cluster')

    g=g.map(plt.hist, feature)
data1.insert(0, 'TENURE', data1['TENURE'], allow_duplicates=True) # just for convience

each_cluster=data1.groupby('cluster').mean().iloc[:, :23]

each_cluster
high=['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',

      'MINIMUM_PAYMENTS']

low=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 

     'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 

     'PURCHASES_TRX','PRC_FULL_PAYMENT', 'TENURE']

top=['CREDIT_LIMIT', 'PURCHASES', 'PAYMENTS']



each_cluster_high=each_cluster[high].T

each_cluster_low=each_cluster[low].T

each_cluster_top=each_cluster[top].T
def render_plot(data, dataLenth, labels, color, facecolor):    

    angles = np.linspace(0, 2*np.pi, dataLenth, endpoint=False)

    data, angles = np.concatenate((data, [data[0]])), np.concatenate((angles, [angles[0]])) # for visualize circle

        

    ax = fig.add_subplot(121, polar=True)# polar: drawing circle

    ax.plot(angles, data, color, linewidth=1)

    ax.fill(angles, data, facecolor=facecolor, alpha=0.1)# fill color

    ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="SimHei")

    ax.set_title("Feature Value of Each Cluster", va='baseline', fontproperties="SimHei")

    ax.grid(True)
fig = plt.figure(figsize=(15,15))

labels = np.array(list(each_cluster_top.index))

render_plot(each_cluster_top.iloc[:,0], len(each_cluster_top.iloc[:,0]), labels,'go-', 'g')

render_plot(each_cluster_top.iloc[:,1], len(each_cluster_top.iloc[:,0]), labels,'bo-', 'b')

render_plot(each_cluster_top.iloc[:,2], len(each_cluster_top.iloc[:,0]), labels,'ro-', 'r')
fig=plt.figure()

fig = plt.figure(figsize=(15,15))

labels = np.array(list(each_cluster_high.index))

render_plot(each_cluster_high.iloc[:,0], len(each_cluster_high.iloc[:,0]), labels,'go-', 'g')

render_plot(each_cluster_high.iloc[:,1], len(each_cluster_high.iloc[:,0]), labels,'bo-', 'b')

render_plot(each_cluster_high.iloc[:,2], len(each_cluster_high.iloc[:,0]), labels,'ro-', 'r')
fig = plt.figure(figsize=(15,15))

labels = np.array(list(each_cluster_low.index))

render_plot(each_cluster_low.iloc[:,0], len(each_cluster_low.iloc[:,0]), labels,'go-', 'g')

render_plot(each_cluster_low.iloc[:,1], len(each_cluster_low.iloc[:,0]), labels,'bo-', 'b')

render_plot(each_cluster_low.iloc[:,2], len(each_cluster_low.iloc[:,0]), labels,'ro-', 'r')
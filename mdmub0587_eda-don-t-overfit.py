%cd '../input/older-dataset-for-dont-overfit-ii-challenge/'
!ls
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
data = pd.read_csv('train.csv')
data.head()
data.shape
data.describe()
nul = 0
for col in data.columns:
    if(data[col].isnull().any()):
        print(col,'has null values')
        nul = 1
if(nul==0):
    print('There is no Null value present')
msno.matrix(data)
sns.set(rc={'figure.figsize':(30,10)})
plt.subplot(1,2,1)
ax = sns.countplot(data['target'])
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(100*p.get_height()/len(data)), (p.get_x()+0.35, p.get_height()+1))
ax.set_title('count plot of Target Distribution')
ax.set_xlabel('Target')
ax.set_ylabel('Count')

plt.subplot(1,2,2)
ax = data['target'].value_counts().plot(kind='pie', colormap='coolwarm')
ax.set_title('Pic chart of Target Distribution')
# https://stackoverflow.com/questions/50940283/show-metrics-like-kurtosis-skewness-on-distribution-plot-using-seaborn-in-pytho
sns.set(rc={'figure.figsize':(30,90)})
fig, ax = plt.subplots(30, 10,sharex='col', sharey='row')
ax = ax.reshape(-1)
for i, col in enumerate(data.columns[2:]):
    g0 = data[data['target']==0.0][col]
    g1 = data[data['target']==1.0][col]
    sns.distplot(g0, label = 'target 0', ax=ax[i], color='b')
    sns.distplot(g1, label = 'target 1', ax=ax[i], color='r')

min_skew, max_skew = 100, -100
min_kurt, max_kurt = 100, -100

for i, x in enumerate(ax):
    skew = data.iloc[:,i+2].skew()
    min_skew = min(min_skew, skew)
    max_skew = max(max_skew, skew)
    x.text(x=0.97, y=0.97, transform=x.transAxes, s="Skewness: %f" % skew,\
        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:poo brown')
    kurt = data.iloc[:,i+2].kurt()
    min_kurt = min(min_kurt, kurt)
    max_kurt = max(max_kurt, kurt)
    x.text(x=0.97, y=0.87, transform=x.transAxes, s="Kurtosis: %f" % kurt,\
        fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
        backgroundcolor='white', color='xkcd:dried blood')
plt.tight_layout()
plt.show()
print('Skewness Range:[{},{}] '.format(min_skew,max_skew))
print('Kurtosis Range:[{},{}] '.format(min_kurt,max_kurt))
sns.set(rc={'figure.figsize':(30,90)})
fig, ax = plt.subplots(30, 10,sharex='col', sharey='row')
fig.text(0.5, 0.12, s='Target', ha='center',)
fig.text(0.1, 0.5, s='Distribution', va='center', rotation='vertical')
ax = ax.reshape(-1)
for i, col in enumerate(data.columns[2:]):
    axe= sns.boxenplot(np.array(data['target']),np.array(data[col]), ax=ax[i])
    ax[i].set_title('Feature '+col)
plt.show()
sns.set(rc={'figure.figsize':(30,15)})
matrix = np.tril(data.drop(['id'], axis=1).corr())
ax = sns.heatmap(data.drop(['id'], axis=1).corr(), cbar_kws= {'orientation': 'horizontal'}, mask=matrix)
correlations = data.drop(['id'], axis=1).corr().unstack().drop_duplicates()
print('Top 10 positive correlated features with target:')
print(correlations["target"].sort_values(ascending=False)[:10])
print('-'*100)
print('Top 10 negative correlated features with target:',)
print(correlations["target"].sort_values(ascending=True)[:10])
sns.set(rc={'figure.figsize':(30,10)})
plt.subplot(1,2,1)
x = correlations["target"].sort_values(ascending=False)[1:10].plot(kind='bar', title='Top 10 Positively correlated features with target')
x.set_xlabel('Features')
x.set_ylabel('Correlation values')

plt.subplot(1,2,2)
x = correlations["target"].sort_values(ascending=True)[:10].plot(kind='bar', title='Top 10 Negatively correlated features with target', color='r')
x.set_xlabel('Features')
x.set_ylabel('Correlation values')
features = list(correlations["target"].sort_values(ascending=False)[:5].index)
ax = sns.pairplot(data[features], hue='target', height=3)
features = list(correlations["target"].sort_values(ascending=True)[:5].index)
ax = sns.pairplot(data[features+['target']], hue='target', height=2.5)
sns.set(rc={'figure.figsize':(20,10)})
pca = PCA(n_components=3, svd_solver='full')
pca_result = pca.fit_transform(data.drop(['id','target'], axis=1).values)
x = pca_result[:,0]
y = pca_result[:,1]
ax = sns.scatterplot(x, y, hue = data['target'], palette=sns.color_palette("hls", 2), legend="full", alpha=1)
ax.set_title('2D visualization')
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax = plt.figure(figsize=(20,10)).gca(projection='3d')
z = pca_result[:,2]
ax.scatter(xs=x, ys=y, zs=z, c=data['target'], cmap='Accent')
ax.set_title('3D visualization')
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()
sns.set(rc={'figure.figsize':(30,15)})
g0 = data[data['target']==0.0][data.columns[2:]]
g1 = data[data['target']==1.0][data.columns[2:]]


for i, fun in enumerate(['mean','std','skew','kurt']):
    plt.subplot(2,2,i+1)
    if(fun=='mean'):
        f0, f1, p0, p1 = g0.mean().mean(), g1.mean().mean(), g0.mean(), g1.mean()
    elif(fun=='std'):
        f0, f1, p0, p1 = g0.std().std(), g1.std().std(), g0.std(), g1.std()
    elif(fun=='skew'):
        f0, f1, p0, p1 = g0.skew().skew(), g1.skew().skew(), g0.skew(), g1.skew()
    elif(fun=='kurt'):
        f0, f1, p0, p1 = g0.kurt().kurt(), g1.kurt().kurt(), g0.kurt(), g1.kurt()

    x = p0.plot(kind = 'hist', alpha=0.5, label='0', color='b')
    x.text(x=0.90, y=0.98, transform=x.transAxes, s="Target 0 {}: {:.4f}".format(fun,f0),\
            fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
            backgroundcolor='white', color='b')
    x = p1.plot(kind = 'hist', alpha=0.5, label='1', color='r')
    x.text(x=0.90, y=0.92, transform=x.transAxes, s="Target 1 {}: {:.4f}".format(fun,f1),\
            fontweight='demibold', fontsize=10, verticalalignment='top', horizontalalignment='right',\
            backgroundcolor='white', color='r')
    plt.title(fun+' Frequency')
    plt.legend()
plt.show()
test = pd.read_csv('test.csv')
test.shape
bf = list()
for i in data.columns[2:]:
    skew = data[i].skew()
    if(skew>0.3 or skew<-0.3):
        bf.append((i,skew))
print('Number of Features for binning: ', len(bf))
print('Features for with their skewness: ', bf)
bf = dict(bf)
# https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b
for col in bf.keys():
    quantile_list = [0, .25, .5, .75, 1.]
    quantiles = data[col].quantile(quantile_list)
    quantile_labels = [0, 1, 2, 3]
    data['Quantile_binning_'+col] = pd.qcut(data[col], q=quantile_list, labels=quantile_labels)
    test['Quantile_binning_'+col] = pd.qcut(test[col], q=quantile_list, labels=quantile_labels)
sns.set(rc={'figure.figsize':(30,35)})

cnt = 1;
for col in bf.keys():
    plt.subplot(len(bf.keys()),2,cnt)
    g0 = data[data['target']==0.0][col]
    g1 = data[data['target']==1.0][col]
    sns.distplot(g0, label = 'target 0', color='b')
    sns.distplot(g1, label = 'target 1', color='r')
    plt.legend()

    plt.subplot(len(bf.keys()),2,cnt+1)
    g0 = data[data['target']==0.0]['Quantile_binning_'+col]
    g1 = data[data['target']==1.0]['Quantile_binning_'+col]
    sns.distplot(g0, label = 'target 0', color='b')
    sns.distplot(g1, label = 'target 1', color='r')
    plt.legend()
    cnt += 2

plt.show()
data.head()
test.head()
fe = ['Quantile_binning_'+f for f in bf.keys()]
fe
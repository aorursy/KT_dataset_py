import numpy as np # linear algebra

import pandas as pd # data processing



# For Visualisation

import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import missingno as msno



# For scaling the data

from sklearn.preprocessing import StandardScaler



# To perform K-means clustering

from sklearn.cluster import KMeans



# To perform PCA

from sklearn.decomposition import PCA,IncrementalPCA



#To perform hierarchical clustering

from scipy.cluster.hierarchy import linkage

from scipy.cluster.hierarchy import dendrogram

from scipy.cluster.hierarchy import cut_tree
#Reading the Dataset 

Country_data=pd.read_csv("/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv")
#Shape of dataset

print("There are {} countries and {} features: ".format(Country_data.shape[0],Country_data.shape[1]))

#Reading the first 5 rows of the dataset

Country_data.head()
Country_data['exports'] = Country_data['exports']*Country_data['gdpp']/100

Country_data['imports'] = Country_data['imports']*Country_data['gdpp']/100

Country_data['health'] = Country_data['health']*Country_data['gdpp']/100

print('After Conversion')

Country_data.head()
msno.matrix(Country_data)

plt.savefig('missing matrix.png')
msno.bar(Country_data)

plt.savefig('missing barplot.png')
print('Null values: \n{}'.format(Country_data.isnull().sum()))

print('\nNaN values: \n{}'.format(Country_data.isna().sum()))
# Checking the datatypes of each variable

Country_data.info()
#finding duplicates

print('There are {} duplicates in dataset'.format(len(Country_data[Country_data.duplicated()])))
# Checking outliers at 25%,50%,75%,90%,95% and 99%

Country_data.describe(percentiles=[.25,.5,.75,.90,.95,.99])
plt.figure(figsize=(15,10))

sns.heatmap(Country_data.corr(),annot=True,cmap='Blues').get_figure().savefig('correlation_heatmap.png')
f, axes = plt.subplots(3, 3, figsize=(15, 15))

s=sns.boxplot(y=Country_data.child_mort,ax=axes[0, 0],color="#FC9803")

axes[0, 0].set_title('Child Mortality Rate')

s=sns.boxplot(y=Country_data.exports,ax=axes[0, 1],color="#FC9803")

axes[0, 1].set_title('Exports')

s=sns.boxplot(y=Country_data.health,ax=axes[0, 2],color="#FC9803")

axes[0, 2].set_title('Health')



s=sns.boxplot(y=Country_data.imports,ax=axes[1, 0],color="#fc9803")

axes[1, 0].set_title('Imports')

s=sns.boxplot(y=Country_data.income,ax=axes[1, 1],color="#fc9803")

axes[1, 1].set_title('Income per Person')

s=sns.boxplot(y=Country_data.inflation,ax=axes[1, 2],color="#fc9803")

axes[1, 2].set_title('Inflation')



s=sns.boxplot(y=Country_data.life_expec,ax=axes[2, 0],color="#fc9803")

axes[2, 0].set_title('Life Expectancy')

s=sns.boxplot(y=Country_data.total_fer,ax=axes[2, 1],color="#fc9803")

axes[2, 1].set_title('Total Fertility')

s=sns.boxplot(y=Country_data.gdpp,ax=axes[2, 2],color="#fc9803")

axes[2, 2].set_title('GDP per Capita')

s.get_figure().savefig('boxplot subplots.png')

plt.show()
pair2=sns.pairplot(Country_data,diag_kind='kde',corner=True,plot_kws=dict(s=7, edgecolor="r", linewidth=1))

pair2.savefig('pairplot.png')
# Droping string feature country name.

features=Country_data.drop('country',1) 



#creating scaler object 

standard_scaler = StandardScaler()

features_scaled = standard_scaler.fit_transform(features)

features_scaled
#creating pca object

pca = PCA(svd_solver='randomized', random_state=42)

# fiting PCA on the dataset

pca.fit(features_scaled)

#checking components

print("{} pca components.\nList of components\n{}".format(pca.n_components_,pca.components_))
#creating variable for no. of components

comp=range(1,pca.n_components_+1)

#cumulative variance of the first 5 principal components

print("The cumulative variance of the first 5 principal components is {}".format(round(pca.explained_variance_ratio_.cumsum()[4],5)))

#plotting barplot for pca components' explained variance ratio

plt.figure(figsize=(10,5))

plt.bar(comp,pca.explained_variance_ratio_)

plt.xticks(comp)

plt.title('PCA components\' Explained Variance ratio',fontsize=20)

plt.xlabel('Number of components',fontsize=15)

plt.ylabel('Explained Variance Ratio',fontsize=15)

plt.savefig('EVR PCA Barplot.png')

plt.show()
%matplotlib inline

fig = plt.figure(figsize = (10,5))

plt.plot(comp,np.cumsum(pca.explained_variance_ratio_),marker='o',markersize=7,color='r')

plt.title('SCREE PLOT',fontsize=25)

plt.xlabel('Number of Components',fontsize=15)

plt.ylabel('Cumulative Explained Variance Ratio',fontsize=15)

plt.vlines(x=5, ymax=1, ymin=0.58, colors="b", linestyles="-")

plt.hlines(y=pca.explained_variance_ratio_.cumsum()[4], xmax=8, xmin=0, colors="b", linestyles="--")

plt.xticks(comp)

plt.savefig('EVR PCA Screeplot.png')

plt.show()
#creating dataframe of first 5 PCA Components

colnames = list(features.columns)

pca_data = pd.DataFrame({ 'Features':colnames,'PC1':pca.components_[0],'PC2':pca.components_[1],'PC3':pca.components_[2],

                      'PC4':pca.components_[3],'PC5':pca.components_[4]})

pca_data
%matplotlib inline

fig = plt.figure(figsize = (10,5))

sns.scatterplot(pca_data.PC1, pca_data.PC2,hue=pca_data.Features,marker='o',s=70)

plt.title('Scatterplot',fontsize=15)

plt.xlabel('Principal Component 1',fontsize=15)

plt.ylabel('Principal Component 2',fontsize=15)

plt.savefig('PCA Scatterplot.png')

plt.show()
#Finally let's go ahead and do dimensionality reduction using the 5 Principal Components

#using Incremental PCA over normal PCA will use less Memory.

ipca = IncrementalPCA(n_components=5)

ipca = ipca.fit_transform(features_scaled)

ipcat = np.transpose(ipca)

pca_data = pd.DataFrame({'PC1':ipcat[0],'PC2':ipcat[1],'PC3':ipcat[2],'PC4':ipcat[3],'PC5':ipcat[4]})

pca_data
#creating list ks for no. of clusters

ks=list(range(1,10))

plt.figure(figsize=(10,5))

ssd = []

#iterating ks values and fitting each value to the kmeans model

for num_clusters in ks:

    model = KMeans(n_clusters = num_clusters, max_iter=50)

    model.fit(pca_data.iloc[:,:5])

    ssd.append(model.inertia_)
plt.figure(figsize = (12,8))

plt.plot(ks,ssd,marker='o',markersize=7)

plt.vlines(x=3, ymax=ssd[-1], ymin=ssd[0], colors="r", linestyles="-")

plt.hlines(y=ssd[2], xmax=9, xmin=1, colors="r", linestyles="--")

plt.title('Elbow Method',fontsize=15)

plt.xlabel('Number of clusters',fontsize=15)

plt.ylabel('Sum of Squared distance',fontsize=15)

plt.savefig('Kmeans elbow.png')

plt.show()
#chosing no. of clusters as 3 and refitting kmeans model

kmeans = KMeans(n_clusters = 3, max_iter=50,random_state = 50)

kmeans.fit(pca_data.iloc[:,:5])

#adding produced labels to pca_data 

pca_data['ClusterID']= pd.Series(kmeans.labels_)

fig = plt.figure(figsize = (12,8))

sns.scatterplot(x='PC1',y='PC2',hue='ClusterID',legend='full',data=pca_data,palette="muted")

plt.title('Categories of countries on the basis of Components')

plt.savefig('Kmeans pca scatter.png')

plt.show()
final_df=pd.merge(Country_data,pca_data.loc[:,'ClusterID'], left_index=True,right_index=True)

final_df.head()
#calculating mean of the required columns(child_mort, income, gdpp) for comparison

Cluster_GDPP=pd.DataFrame(final_df.groupby(["ClusterID"]).gdpp.mean())

Cluster_child_mort=pd.DataFrame(final_df.groupby(["ClusterID"]).child_mort.mean())

Cluster_income=pd.DataFrame(final_df.groupby(["ClusterID"]).income.mean())

K_mean_df = pd.concat([Cluster_GDPP,Cluster_child_mort,Cluster_income], axis=1)

K_mean_df
f, axes = plt.subplots(1, 3, figsize=(20,5))



s=sns.scatterplot(x='child_mort',y='gdpp',data=K_mean_df,hue=K_mean_df.index,palette='Set1',ax=axes[0])

s=sns.scatterplot(x='income',y='gdpp',data=K_mean_df,hue=K_mean_df.index,palette='Set1',ax=axes[1])

s=sns.scatterplot(x='child_mort',y='income',data=K_mean_df,hue=K_mean_df.index,palette='Set1',ax=axes[2])

s.get_figure().savefig('comparison scatterplots.png')
K_mean_df.rename(index={0: 'Developing'},inplace=True)

K_mean_df.rename(index={1: 'Developed'},inplace=True)

K_mean_df.rename(index={2: 'Under-developed'},inplace=True)
f, axes = plt.subplots(1, 3, figsize=(20,7))



s=sns.barplot(x=K_mean_df.index,y='gdpp',data=K_mean_df,ax=axes[0])

axes[0].set_title('GDP per capita',fontsize=10)



s=sns.barplot(x=K_mean_df.index,y='income',data=K_mean_df,ax=axes[1])

axes[1].set_title('income per person')



s=sns.barplot(x=K_mean_df.index,y='child_mort',data=K_mean_df,ax=axes[2])

axes[2].set_title('Child Mortality Rate')



s.get_figure().savefig('comparison subplots.png')

plt.show()
for i,cluster in enumerate(final_df.ClusterID):

    if(cluster==0):

        final_df.loc[i,'Clustered Countries']='Developing'

    elif(cluster==1):

        final_df.loc[i,'Clustered Countries']='Developed'

    else:

        final_df.loc[i,'Clustered Countries']='Under-Developed'

        

f, axes = plt.subplots(3, 1, figsize=(15,20))

s=sns.scatterplot(x='child_mort',y='gdpp',hue='Clustered Countries',legend='full',data=final_df,palette=sns.color_palette("hls", 3),ax=axes[0])

axes[0].set_title('GDPP x Child Mortality Rate',fontsize=15)

s=sns.scatterplot(x='income',y='gdpp',hue='Clustered Countries',legend='full',data=final_df,palette=sns.color_palette("hls", 3),ax=axes[1])

axes[1].set_title('GDPP x Income per person',fontsize=15)

s=sns.scatterplot(x='child_mort',y='income',hue='Clustered Countries',legend='full',data=final_df,palette=sns.color_palette("hls", 3),ax=axes[2])

axes[2].set_title('Income per person x Child Mortality Rate',fontsize=15)

s.get_figure().savefig('scatterplot subplots.png')

plt.show()
f, axes = plt.subplots(1, 3, figsize=(25,7))

sns.boxplot(x='Clustered Countries',y='gdpp',data=final_df,ax=axes[0])

axes[0].set_title('GDP per capita',fontsize=15)

sns.boxplot(x='Clustered Countries',y='income',data=final_df,ax=axes[1])

axes[1].set_title('Income per person',fontsize=15)

sns.boxplot(x='Clustered Countries',y='child_mort',data=final_df,ax=axes[2])

axes[2].set_title('Child Mortality rate',fontsize=15)

s.get_figure().savefig('comparison subplots.png')

plt.show()
f, axes = plt.subplots(3, 1, figsize=(25,35))



gdp_developed=final_df[final_df['Clustered Countries']=='Developed'].sort_values(by='gdpp',ascending=False)

s=sns.barplot(x='country',y='gdpp',data=gdp_developed,palette='Set1',ax=axes[0])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[0].set_title('Developed Countries GDPP Ranking',fontsize=15)



income_developed=final_df[final_df['Clustered Countries']=='Developed'].sort_values(by='income',ascending=False)

s=sns.barplot(x='country',y='income',data=income_developed,palette='Set2',ax=axes[1])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[1].set_title('Developed Countries Income Ranking',fontsize=15)



child_developed=final_df[final_df['Clustered Countries']=='Developed'].sort_values(by='child_mort')

s=sns.barplot(x='country',y='child_mort',data=child_developed,palette='Set3',ax=axes[2])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[2].set_title('Developed Countries Child mortality Ranking',fontsize=15)



s.get_figure().savefig('DEVELOPED Countries rankings.png')

plt.show()
#top 10 developed countries based on GDPP

print('top 10 developed countries based on high GDPP\n')

for countries in gdp_developed.country[:10]:

    print(countries)



#top 10 developed countries based on income

print('\ntop 10 developed countries based on high income\n')

for countries in income_developed.country[:10]:

    print(countries)

    

#top 10 developed countries based on childmort

print('\ntop 10 developed countries based on child low mortality\n')

for countries in child_developed.country[:10]:

    print(countries)
f, axes = plt.subplots(3, 1, figsize=(25,70))



gdp_developing=final_df[final_df['Clustered Countries']=='Developing'].sort_values(by='gdpp',ascending=False)

s=sns.barplot(x='country',y='gdpp',data=gdp_developing,palette='Set1',ax=axes[0])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[0].set_title('Developing Countries GDPP Ranking',fontsize=15)



income_developing=final_df[final_df['Clustered Countries']=='Developing'].sort_values(by='income',ascending=False)

s=sns.barplot(x='country',y='income',data=income_developing,palette='Set2',ax=axes[1])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[1].set_title('Developing Countries Income Ranking',fontsize=15)



child_developing=final_df[final_df['Clustered Countries']=='Developing'].sort_values(by='child_mort')

s=sns.barplot(x='country',y='child_mort',data=child_developing,palette='Set3',ax=axes[2])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[2].set_title('Developing Countries Child mortality Ranking',fontsize=15)



s.get_figure().savefig('DEVELOPING Countries rankings.png')

plt.show()
f, axes = plt.subplots(3, 1, figsize=(25,70))



gdp_under=final_df[final_df['Clustered Countries']=='Under-Developed'].sort_values(by='gdpp',ascending=False)

s=sns.barplot(x='country',y='gdpp',data=gdp_under,palette='Set1',ax=axes[0])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[0].set_title('Under-Developed Countries GDPP Ranking',fontsize=15)



income_under=final_df[final_df['Clustered Countries']=='Under-Developed'].sort_values(by='income',ascending=False)

s=sns.barplot(x='country',y='income',data=income_under,palette='Set2',ax=axes[1])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[1].set_title('Under-Developed Countries Income Ranking',fontsize=15)



child_under=final_df[final_df['Clustered Countries']=='Under-Developed'].sort_values(by='child_mort')

s=sns.barplot(x='country',y='child_mort',data=child_under,palette='Set3',ax=axes[2])

s.set_xticklabels(s.get_xticklabels(),rotation=90)

axes[2].set_title('Under-Developed Countries Child mortality Ranking',fontsize=15)



s.get_figure().savefig('Under-Developed Countries rankings.png')

plt.show()
#top 10 developed countries based on high GDPP

print('top 10 developed countries based on low GDPP\n')

for countries in gdp_under.country[:10]:

    print(countries)



#top 10 developed countries based on high income

print('\ntop 10 developed countries based on low income\n')

for countries in income_under.country[:10]:

    print(countries)

    

#top 10 developed countries based on childmort

print('\ntop 10 developed countries based on child high mortality\n')

for countries in child_under.country[:10]:

    print(countries)
k_needy=Country_data[Country_data['gdpp']<=1909]

k_needy=k_needy[k_needy['child_mort']>= 92]

k_needy=k_needy[k_needy['income']<= 3897]

k_needy=pd.merge(k_needy,pca_data.loc[:,'ClusterID'],left_index=True,right_index=True)

k_needy=k_needy.sort_values(by=['gdpp','income','child_mort'],ascending=[True,True,False])

#Top 10 countries having dire need of aid based on overall conditions

print('\nTop 10 countries having dire need of aid based on overall conditions\n')

for countries in k_needy.country[:10]:

    print(countries)
mergings=linkage(pca_data.iloc[:,:5],method='single',metric='euclidean')

plt.figure(figsize=(25,7))

dn=dendrogram(mergings)

plt.savefig('Single Linkage.png')
mergings=linkage(pca_data.iloc[:,:5],method='complete',metric='euclidean')

plt.figure(figsize=(25,7))

dn=dendrogram(mergings)

plt.savefig('Complete Linkage.png')
h_clusters=cut_tree(mergings,n_clusters=3)

pca_data['H_ClusterID']=h_clusters.reshape(-1)

pca_data.head()
#value counts of cluster ids

pca_data['H_ClusterID'].value_counts()
final_df=pd.merge(final_df,pca_data.loc[:,'H_ClusterID'], left_index=True,right_index=True)

final_df.head()
Cluster_GDPP_H=pd.DataFrame(final_df.groupby(["H_ClusterID"]).gdpp.mean())

Cluster_child_mort_H=pd.DataFrame(final_df.groupby(["H_ClusterID"]).child_mort.mean())

Cluster_income_H=pd.DataFrame(final_df.groupby(["H_ClusterID"]).income.mean())

H_mean_df = pd.concat([Cluster_GDPP_H,Cluster_child_mort_H,Cluster_income_H], axis=1)

H_mean_df
K_mean_df
H_needy=Country_data[Country_data['gdpp']<=2330]

H_needy=H_needy[H_needy['child_mort']>= 130]

H_needy=H_needy[H_needy['income']<= 3897.35]

H_needy=pd.merge(H_needy,pca_data.loc[:,'H_ClusterID'],left_index=True,right_index=True)

H_needy=H_needy.sort_values(by=['gdpp','income','child_mort'],ascending=[True,True,False])

#top 10 developed countries based on childmort

print('Countries having dire need of aid based on Hierarchical clustering are as follows:\n')

for countries in H_needy.country:

    print(countries)
%pylab

#begin import from course

%matplotlib inline 

#%matplotlib notebook



import math

import scipy.stats as stats

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

#end import from course



from collections import defaultdict

from scipy.stats.stats import pearsonr

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import parallel_coordinates



# Any results you write to the current directory are saved as output.



df = pd.read_csv("../input/hr-analytics/HR_comma_sep.csv")

dfb = df



level=df['satisfaction_level'].values

time=df['last_evaluation'].values

projects=df['number_project'].values

hours=df['average_montly_hours'].values

years=df['time_spend_company'].values

accident=df['Work_accident'].values

left=df['left'].values

promotion=df['promotion_last_5years'].values

area=df['sales'].values

salary=df['salary'].values



figsize(10, 7)#graphs size
#correlazione (conteggi con percentuali) compresi i binari



temp = pd.crosstab(df['left'], df['sales'],margins=True, normalize="columns")

S=temp.loc[0] #i dipendenti che non hanno lasciato

outlier=S[(S-S.mean()).abs()>1*S.std()]

print(outlier)
#median

mdn_level=df['satisfaction_level'].median()

mdn_time=df['last_evaluation'].median()

mdn_projects=df['number_project'].median()

mdn_hours=df['average_montly_hours'].median()

mdn_years=df['time_spend_company'].median()



print(mdn_level)

print(mdn_time)

print(mdn_projects)

print(mdn_hours)

print(mdn_years)
#mean

mn_level=df['satisfaction_level'].mean()

mn_time=df['last_evaluation'].mean()

mn_projects=df['number_project'].mean()

mn_hours=df['average_montly_hours'].mean()

mn_years=df['time_spend_company'].mean()



print(mn_level)

print(mn_time)

print(mn_projects)

print(mn_hours)

print(mn_years)
#mode

mode_level=df['satisfaction_level'].mode()

mode_time=df['last_evaluation'].mode()

mode_projects=df['number_project'].mode()

mode_hours=df['average_montly_hours'].mode()

mode_years=df['time_spend_company'].mode()

mode_accident=df['Work_accident'].mode()

mode_left=df['left'].mode()

mode_promotion=df['promotion_last_5years'].mode()

mode_area=df['sales'].mode()

mode_salary=df['salary'].mode()



print(mode_level)

print(mode_time)

print(mode_projects)

print(mode_hours)

print(mode_years)

print(mode_accident)

print(mode_left)

print(mode_promotion)

print(mode_area)

print(mode_salary)
#standard deviation

sd_level=df['satisfaction_level'].std()

sd_time=df['last_evaluation'].std()

sd_projects=df['number_project'].std()

sd_hours=df['average_montly_hours'].std()

sd_years=df['time_spend_company'].std()



print(sd_level)

print(sd_time)

print(sd_projects)

print(sd_hours)

print(sd_years)
#BOXPLOTS

fig = plt.figure()

fig_dims = (3, 2)

plt.subplot2grid(fig_dims, (0, 0))

plt.boxplot(df['number_project'])

plt.subplot2grid(fig_dims, (0, 1))

plt.boxplot(df['average_montly_hours'])

plt.subplot2grid(fig_dims, (1, 0))

plt.boxplot(df['time_spend_company'])

plt.subplot2grid(fig_dims, (1, 1))

plt.boxplot(df['satisfaction_level'])

plt.subplot2grid(fig_dims, (2, 0))

plt.boxplot(df['last_evaluation'])
#NORMALIZZAZIONE

if(dtype(df['number_project'])!=float64):#guardia per evitare di rieseguire la normalizzazione

    

    min_lvl=min(level)

    max_lvl=max(level)

    min_evl=min(time)

    max_evl=max(time)

    min_prj=min(projects)

    max_prj=max(projects)

    min_h=min(hours)

    max_h=max(hours)

    min_t=min(years)

    max_t=max(years)



    df['number_project']=df['number_project'].astype(float64)

    df['average_montly_hours']=df['average_montly_hours'].astype(float64)

    df['time_spend_company']=df['time_spend_company'].astype(float64)



    for i in range(0,14999):

        #normalization min-max

        df['satisfaction_level'].values[i]=(float(df['satisfaction_level'].values[i])-min_lvl)/(max_lvl-min_lvl)

        df['last_evaluation'].values[i]=(float(df['last_evaluation'].values[i])-min_evl)/(max_evl-min_evl)

        df['number_project'].values[i]=(float(df['number_project'].values[i])-min_prj)/(max_prj-min_prj)

        df['average_montly_hours'].values[i]=(float(df['average_montly_hours'].values[i])-min_h)/(max_h-min_h)

        df['time_spend_company'].values[i]=(float(df['time_spend_company'].values[i])-min_t)/(max_t-min_t)

        # transformation of the continuous values of attribute into a few discrete values

        if (df['salary'].values[i] == 'low'):

            df['salary'].values[i]=0

        elif (df['salary'].values[i] == 'medium'):

            df['salary'].values[i]=0.5

        else:

            df['salary'].values[i]=1



    df['salary']=pd.to_numeric(df['salary'])





df_corr=df.drop(['Work_accident','left','promotion_last_5years','sales'],axis=1)
#correlazione Pearson

df_corr.corr()
#correlazione Spearman

df_corr.corr(method='spearman')
#differenza pearson-spearman

%pylab inline

plt.figure(figsize=(10, 10))

corr=df_corr.corr()#-df_corr.corr(method='spearman')

plt.imshow(corr, cmap='RdYlBu', interpolation='none', aspect='auto')

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation='vertical')

plt.yticks(range(len(corr)), corr.columns);

plt.suptitle('Pairwise Correlations Heat Map', fontsize=15, fontweight='bold')

plt.show()
#PARALLEL COORDINATES

plt.figure() 

parallel_coordinates(df, 'sales')
def hist_PDF( dim,num_bins=0):

    tot_values=df[dim].size

    if(num_bins == 0):

        num_bins=int(1+ceil(math.log(tot_values,2)))#sturges' rule

    n, bins, patches = plt.hist(df[dim], num_bins, color='green')#, title=dim)

    #df[dim].hist(bins=num_bins)



    # Plot the PDF.

    time=df[dim].values

    xmin=min(time)

    xmax=max(time)

    mn_level=df[dim].mean()

    sd_time=df[dim].std()

    x = np.linspace(xmin, xmax, 100)

    from scipy.stats import norm

    p = norm.pdf(x, mn_level, sd_time)*tot_values/num_bins

    plt.plot(x, p, 'k', linewidth=2)

    

    cm = plt.cm.get_cmap('RdYlBu_r')

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # scale values to interval [0,1]

    col = bin_centers - min(bin_centers)

    col /= max(col)



    for c, p in zip(col, patches):

        plt.setp(p, 'facecolor', cm(c))

        

    plt.suptitle(dim, fontsize=15, fontweight='bold')

    return

'''

#codice per disegnare bimodale

def gauss(x,mu,sigma,A):

    return A*exp(-(x-mu)**2/2/sigma**2)



def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):

    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)



dim='average_montly_hours'

tot_values=df[dim].size

#if(num_bins == 0):

num_bins=int(1+ceil(math.log(tot_values,2)))#sturges' rule

n, bins, patches = plt.hist(df[dim], num_bins, color='green')

#df[dim].hist(bins=num_bins)



# Plot the PDF.

time=df[dim].values

xmin=min(time)

xmax=max(time)

mn_level=df[dim].mean()

sd_time=df[dim].std()

x = np.linspace(xmin, xmax, 100)

from scipy.stats import norm

a = norm.pdf(x, 0.26, 0.11)

b = norm.pdf(x, 0.75, 0.125)

p = (a + b + 8*a*b)*tot_values/(2*num_bins)

plt.plot(x, p, 'k', linewidth=2)



cm = plt.cm.get_cmap('RdYlBu_r')

bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]

col = bin_centers - min(bin_centers)

col /= max(col)



for c, p in zip(col, patches):

    plt.setp(p, 'facecolor', cm(c))

'''
hist_PDF('satisfaction_level')
hist_PDF('last_evaluation')
hist_PDF('number_project',6)
hist_PDF('average_montly_hours')
hist_PDF('time_spend_company',9)
df['Work_accident'].value_counts().plot(kind='bar', title='accident')
df['left'].value_counts().plot(kind='bar', title='left')
df['promotion_last_5years'].value_counts().plot(kind='bar', title='promotion')
df['sales'].value_counts().plot(kind='bar', title='area')
df['salary'].value_counts().plot(kind='bar', title='salary')
#BOXPLOT SENZA OUTLIERS (time_spend_company)

df2=years

temp=[]

for i in range(len(df2)):

    if df2[i]<6: 

        temp.append(df2[i])

temp_new = pd.Series( (v for v in temp) )

plt.boxplot(temp_new)
#ISTOGRAMMA SENZA OUTLIERS (time_spend_company)

tot_values=temp_new.size

num_bins=4#int(1+ceil(math.log(tot_values,2)))#sturges' rule

n, bins, patches = plt.hist(temp_new, num_bins, color='green')



min_y2=min(temp_new)

max_y2=max(temp_new)

mn_ty2=temp_new.mean()

sd_y2=temp_new.std()

x = np.linspace(min_y2, max_y2, 100)



from scipy.stats import norm

p = (norm.pdf(x, mn_ty2, sd_y2)*tot_values*(max_y2-min_y2)/num_bins)#+xmin

plt.plot(x, p, 'k', linewidth=2)





cm = plt.cm.get_cmap('RdYlBu_r')

bin_centers = 0.5 * (bins[:-1] + bins[1:])

# scale values to interval [0,1]

col = bin_centers - min(bin_centers)

col /= max(col)



for c, p in zip(col, patches):

    plt.setp(p, 'facecolor', cm(c))

#3D SCATTER

dim1='satisfaction_level'

dim2='last_evaluation'



temp1=array([pd.to_numeric(df.groupby([dim1]).count().index).values,]*len(pd.to_numeric(df[dim2].value_counts())))

temp2=array([pd.to_numeric(df.groupby([dim2]).count().index).values,]*len(pd.to_numeric(df[dim1].value_counts()))).transpose()

temp3= pd.crosstab(df[dim1], df[dim2]).iloc[:,:]



threedee = plt.figure().gca(projection='3d')

threedee.set_xlabel(dim1)

threedee.set_ylabel(dim2)

threedee.set_zlabel('count')

threedee.scatter(temp1,temp2,temp3)

plt.show()
#DF_CLUS inizialization

from sklearn.metrics import *

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.cluster import AgglomerativeClustering

from sklearn.neighbors import kneighbors_graph

from scipy.stats import mode

from scipy.cluster.hierarchy import linkage, dendrogram

from scipy.spatial.distance import pdist, squareform



def clust3D (df_val, dim1, dim2,dim3,  n_clust):

    kmeans = KMeans(n_clusters = n_clust)

    cluster_data = df_val.values

    kmeans.fit(cluster_data)

    ax = Axes3D(plt.figure(), rect=[0, 0, .95, 1], elev=48, azim=-134)

    ax.set_xlim([0,1])

    ax.set_ylim([0,1])

    ax.set_zlim([0,1])

    ax.scatter(df_val[dim1],df_val[dim2],df_val[dim3],c=kmeans.labels_)

    ax.set_xlabel(dim1)

    ax.set_ylabel(dim2)

    ax.set_zlabel(dim3)

    ax.set_title('silhouette:'+ str(silhouette_score(df_val, kmeans.labels_)), fontsize=15, fontweight='bold') 

#The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 

#The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster 

#and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value,

#then the clustering configuration may have too many or too few clusters.

    return kmeans



def hstack(lab,dim):

    df_lab = df_clus.copy(deep=True)

    df_lab['Labels'] = lab

    hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

    temp = pd.crosstab(df[dim], df_lab['Labels'])

    #temp_g = temp.div(temp.sum(1).astype(float), axis=0)

    temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

    plt.xlabel(dim)

    plt.ylabel('Cluster Label')

    plt.show()





df_clus=df.drop(['Work_accident','left','promotion_last_5years','sales','salary'],axis=1)

cleft = df_clus.loc[df['left'] == 1]#pd.Series( (v for v in cl_left) )

cstay = df_clus.loc[df['left'] == 0]#pd.Series( (v for v in cl_stay) )
# kmeans silhouette-sse

sse_list = list() #the sum of the squared distance between each member of the cluster and its centroid

sil_list = list()

max_k = 12



for k in range(2,max_k+1):

    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, max_iter=100)

    kmeans.fit(df_clus.values)

    sse_list.append(kmeans.inertia_)

    sil_list.append(silhouette_score(df_clus.values, kmeans.labels_))#*sse_list[0])



t=range(2,max_k+1)

s1 = sse_list

s2 = sil_list

fig, ax1 = plt.subplots()



ax1.plot(t, s1, 'b-')

ax1.set_xlabel('k')

ax1.set_ylabel('SSE', color='b')

ax1.set_ylim(bottom=0)

ax1.tick_params('y', colors='b')



ax2 = ax1.twinx()

ax2.plot(t, s2, 'r-')

ax2.set_ylabel('silhouette', color='r')

ax2.set_ylim([0,1])

ax2.tick_params('y', colors='r')



fig.tight_layout()

plt.show()
print(sse_list)

print(sil_list)
km=clust3D(df_clus,

        'satisfaction_level',

        'last_evaluation',#dim 1 to plot

        #'number_project',

        'average_montly_hours',

        #'time_spend_company',

        12)#k, number of clusters

lab=km.labels_

km.cluster_centers_
#distribuzione dell'attibuto sul totale

hstack(lab,

    #'last_evaluation'

    #'satisfaction_level'

    #'number_project'

    #'average_montly_hours'

    #'time_spend_company'

    #'Work_accident'

    #'promotion_last_5years'

    'left'

    #'sales'

    #'salary'

    )
#codice per differenziare (e descrivere) i vari clusters

others = df_clus

others = others.loc[others['Labels'] !=10]#1left, low time_company, high last

others = others.loc[others['Labels'] !=5]#1left, low time_company

others = others.loc[others['Labels'] !=6]#outliers time_company

others = others.loc[others['Labels'] !=3]#high project,low sat, high average

#others = others.loc[others['Labels'] !=9]#high last

others = others.loc[others['Labels'] !=1]#1left,high sat, high last, low time_company

others = others.loc[others['Labels'] !=4]#high last, low time_company

#others = others.loc[others['Labels'] !=0]#low last, low time_company

others = others.loc[others['Labels'] !=7]#low sat

#others = others.loc[others['Labels'] !=8]#low last, low time_company, low average, low project

#dim='average_montly_hours'

dim='left'

lab = [x for x in lab if x !=1 and x !=4 and x !=5]

hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

temp = pd.crosstab(df[dim], others['Labels'])

#temp_g = temp.div(temp.sum(1).astype(float), axis=0)

temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

plt.xlabel(dim)

plt.ylabel('Cluster Label')

plt.show()
km=clust3D(cleft,#dims to drop

        #'last_evaluation',

        'satisfaction_level',

        'number_project',

        'average_montly_hours',

        #'time_spend_company',

        3)#k, number of clusters

lab1=km.labels_
km=clust3D(cstay,#dims to drop

        #'last_evaluation',

        'satisfaction_level',

        'number_project',

        #'average_montly_hours',

        'time_spend_company',

        10)#k, number of clusters

lab2=km.labels_
# single linkage and dendogram using scipy

data_dist = pdist(df_clus, metric='euclidean')

data_link = linkage(data_dist, method='complete', metric='euclidean')#anche single

res = dendrogram(data_link,no_labels=True)

col = 0

lab = array('i')

for i in range(0,len(res['ivl'])-1):

    if i>0 and res['color_list'][i]!=res['color_list'][i-1]:

        col=col+1

    lab[int(res['ivl'][i])]=col

res = dendrogram(data_link,no_labels=True)
#see hierarchical

dim1='satisfaction_level'

dim2='last_evaluation'

dim3='average_montly_hours'

ax = Axes3D(plt.figure(), rect=[0, 0, .95, 1], elev=48, azim=-134)

ax.set_xlim([0,1])

ax.set_ylim([0,1])

ax.set_zlim([0,1])

ax.set_xlabel(dim1)

ax.set_ylabel(dim2)

ax.set_zlabel(dim3)

ax.set_title('silhouette:'+ str(silhouette_score(df_clus, lab)), fontsize=15, fontweight='bold')

ax.scatter(df_clus[dim1],df_clus[dim2],df_clus[dim3],c=lab)
# density based clustering



dbscan = DBSCAN(eps=0.19, min_samples=75, metric='euclidean')

dbscan.fit(df_clus)



#hist, bins = np.histogram(dbscan.labels_, bins=range(-1, len(set(dbscan.labels_)) + 1))

dim1='satisfaction_level'

dim2='last_evaluation'

dim3='average_montly_hours'

ax = Axes3D(plt.figure(), rect=[0, 0, .95, 1], elev=48, azim=-134)

ax.set_xlim([0,1])

ax.set_ylim([0,1])

ax.set_zlim([0,1])

ax.set_xlabel(dim1)

ax.set_ylabel(dim2)

ax.set_zlabel(dim3)

ax.set_title('silhouette:'+ str(silhouette_score(df_clus, dbscan.labels_)), fontsize=15, fontweight='bold')

ax.scatter(df_clus[dim1],df_clus[dim2],df_clus[dim3],c=dbscan.labels_) 

lab=dbscan.labels_

#min_samples 250 eps 0.2

#min_samples 200 eps 0.2

#min_samples 100 eps 0.17

#min_samples 75 eps 0.19

#min_samples 10 eps 0.1727

#min_samples 9 eps 0.1695

#stima di min_samples

min_samples=5

data_dist = squareform(pdist(df_clus, metric='euclidean'))

kth_neighbours = []

for i in range (0, len(data_dist)):

    kth_neighbours.append(np.partition(data_dist[i],min_samples)[min_samples])

plt.plot(range(0,14999),sorted(kth_neighbours), 'k', linewidth=2)
#stima di eps

eps=0.2

data_dist = squareform(pdist(df_clus, metric='euclidean'))

k_neighbours = []

for i in range (0, len(data_dist)):

    k_neighbours.append(len(numpy.where(data_dist[i]<eps)[0]))

plt.plot(range(0,14999),sorted(k_neighbours), 'k', linewidth=2)
import fim

from fim import apriori



#http://www.borgelt.net/pyfim.html



min_lvl=min(level)

max_lvl=max(level)

bin_lvl= 10



min_evl=min(time)

max_evl=max(time)

bin_evl=10



min_h=min(hours)

max_h=max(hours)

bin_h=10



baskets = defaultdict(list)



for i in len(dfb):

    

    #baskets[i].append(item_id)

    binsize=(max_lvl-min_lvl)/bin_lvl

    binnumber=int((dfb['satisfaction_level'][i]-min_lvl)/binsize)    

    baskets[i].append(str(binnumber*binsize+min_lvl)+'-'+str((binnumber+1)*binsize+min_lvl)+'_S')    

    

    binsize=(max_evl-min_evl)/bin_evl

    binnumber=int((dfb['last_evaluation'][i]-min_evl)/binsize)    

    baskets[i].append(str(binnumber*binsize+min_evl)+'-'+str((binnumber+1)*binsize+min_evl)+'_E')    

    

    binsize=(max_h-min_h)/bin_h

    binnumber=int((dfb['average_montly_hours'][i]-min_h)/binsize)    

    baskets[i].append(str(binnumber*binsize+min_h)+'-'+str((binnumber+1)*binsize+min_h)+'_A')

    

    baskets[i].append(str(dfb['number_project'][i])+'_N')

    baskets[i].append(str(dfb['time_spend_company'][i])+'_T')

    

    baskets[i].append(str(dfb['Work_accident'][i])+'_W')

    baskets[i].append(str(dfb['left'][i])+'_L')

    baskets[i].append(str(dfb['promotion_last_5years'][i])+'_P')

                      

    baskets[i].append(dfb['sales'][i])

    baskets[i].append(dfb['salary'][i])

    

baskets_lists = [b for b in baskets.values()]



itemsets = apriori(baskets_lists[:100], supp=2, zmin=2, target='a') 
#CLASSIFICATION



import random

from sklearn import tree

from sklearn import metrics

from sklearn.tree import export_graphviz

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import confusion_matrix

from sklearn.cross_validation import train_test_split

from sklearn import cross_validation

from sklearn.cross_validation import cross_val_score

#from sklearn import neighbors

#from sklearn import linear_model

#from sklearn.naive_bayes import GaussianNB

#from sklearn import svm

#Convert the DataFrame to a numpy array:

#train_data = df['left'].values



# Training data features, skip the first column 'Survived'

train_features = df.drop(['left', 'sales'],axis=1).values



# 'Survived' column values

train_target = df['left'].values





clf = tree.DecisionTreeClassifier(criterion='gini', splitter='best', 

                                  max_depth=None, 

                                  min_samples_split=2, min_samples_leaf=2)

clf = clf.fit(train_features, train_target)



import pydotplus 

from IPython.display import Image  



dot_data = tree.export_graphviz(clf, out_file=None, 

                         feature_names=list(df.columns[1:]),  

                         class_names=['Not left', 'left'],  

                         filled=True, rounded=True,  

                         special_characters=True)  

graph = pydotplus.graph_from_dot_data(dot_data)  

Image(graph.create_png())
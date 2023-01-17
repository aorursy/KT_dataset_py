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

dfb = df.copy(deep=True)

dfk = df.copy(deep=True)



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



lab=[]

first = True



def clust3D (df_val, dim1, dim2,dim3,  n_clust):

    global lab

    kmeans = KMeans(n_clusters = n_clust)

    cluster_data = df_val.values

    kmeans.fit(cluster_data)

    ax = Axes3D(plt.figure(), rect=[0, 0, .95, 1], elev=48, azim=-134)

    ax.set_xlim([0,1])

    ax.set_ylim([0,1])

    ax.set_zlim([0,1])

    lab=kmeans.labels_

    ax.scatter(df_val[dim1],df_val[dim2],df_val[dim3],c=lab)

    ax.set_xlabel(dim1)

    ax.set_ylabel(dim2)

    ax.set_zlabel(dim3)

    ax.set_title('silhouette:'+ str(silhouette_score(df_val, lab)), fontsize=15, fontweight='bold') 

#The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). 

#The silhouette ranges from −1 to +1, where a high value indicates that the object is well matched to its own cluster 

#and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value,

#then the clustering configuration may have too many or too few clusters.   

    return 



def hstack(dim):

    global lab

    df_lab = df_clus.copy(deep=True)

    df_lab['Labels'] = lab

    hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

    temp = pd.crosstab(df[dim], df_lab['Labels'])

    #temp_g = temp.div(temp.sum(1).astype(float), axis=0)

    temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

    plt.xlabel(dim)

    plt.ylabel('Cluster Label')

    plt.show()

    return



def dbscan3d(par_df,par_eps,par_minpts,dim1,dim2,dim3):

    global lab

    dbscan = DBSCAN(eps=par_eps, min_samples=par_minpts, metric='euclidean')

    dbscan.fit(par_df)

    ax = Axes3D(plt.figure(), rect=[0, 0, .95, 1], elev=48, azim=-134)

    ax.set_xlim([0,1])

    ax.set_ylim([0,1])

    ax.set_zlim([0,1])

    ax.set_xlabel(dim1)

    ax.set_ylabel(dim2)

    ax.set_zlabel(dim3)

    lab=dbscan.labels_

    #ax.set_title('silhouette:'+ str(silhouette_score(df_clus, lab)), fontsize=15, fontweight='bold')

    ax.scatter(df_clus[dim1],df_clus[dim2],df_clus[dim3],c=lab)

    return



def dendo(df,met,link,threshold):

    global lab

    data_dist = pdist(df, metric=met)

    data_link = linkage(data_dist, method=link, metric=met)#anche single

    res = dendrogram(data_link,no_labels=True,color_threshold=threshold*max(data_link[:,2]))

    col = 0

    lab = zeros(len(res['ivl']))

    for i in range(0,len(res['ivl'])-1):

        if i>0 and res['color_list'][i]!=res['color_list'][i-1]:

            col=col+1

        lab[int(res['ivl'][i])]=col

    res = dendrogram(data_link,no_labels=True,color_threshold=threshold*max(data_link[:,2]))

    return





df_clus=df.drop(['Work_accident','left','promotion_last_5years','sales','salary'],axis=1)

cleft = df_clus.loc[df['left'] == 1]#pd.Series( (v for v in cl_left) )

cstay = df_clus.loc[df['left'] == 0]#pd.Series( (v for v in cl_stay) )
# kmeans silhouette-sse

sse_list = list() #the sum of the squared distance between each member of the cluster and its centroid

sil_list = list()

max_k = 20



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
clust3D(df_clus,

        'satisfaction_level',

        'last_evaluation',#dim 1 to plot

        #'number_project',

        'average_montly_hours',

        #'time_spend_company',

        8)#k, number of clusters
#distribuzione dell'attibuto sul totale

hstack(

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
others = dfk

others['Labels'] = lab

mean = others.loc[others['Labels'] == 4].mean()

std = others.loc[others['Labels'] == 4].std()

print(len(others.loc[others['Labels'] == 4]))

print(mean)

print(std)
#codice per differenziare (e descrivere) i vari clusters

others = df_clus.copy(deep=True)

others['Labels'] = lab



others = others.loc[others['Labels'] !=2]#1left medium time_company, high last, low sat

others = others.loc[others['Labels'] !=4]#1left low time_company, low last, medium sat

others = others.loc[others['Labels'] !=0]#1left low time_company, high last, high sat

others = others.loc[others['Labels'] !=7]# outliers time_company, flat last, medhigh sat

others = others.loc[others['Labels'] !=1]# flat time_company, flat last, low sat, high project

others = others.loc[others['Labels'] !=3]# low time_company, high last, medhigh sat

#others = others.loc[others['Labels'] !=1]# low time_company, low last, medhigh sat, low average, low project

#others = others.loc[others['Labels'] !=6]# low time_company, medlow last, medhigh sat



#dim='satisfaction_level'



dim='last_evaluation'

#dim='number_project'

#dim='average_montly_hours'

#dim='time_spend_company'

#lab = [x for x in lab if x !=1 and x !=4 and x !=5]

hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

temp = pd.crosstab(df[dim], others['Labels'])

#temp_g = temp.div(temp.sum(1).astype(float), axis=0)

temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

plt.xlabel(dim)

plt.ylabel('Cluster Label')

plt.show()
clust3D(cleft,#dims to drop

        #'last_evaluation',

        'satisfaction_level',

        'number_project',

        'average_montly_hours',

        #'time_spend_company',

        3)#k, number of clusters
#PER SVIRGIO

clust3D(cstay,#dims to drop

        'last_evaluation',

        'satisfaction_level',

        #'number_project',

        'average_montly_hours',

        #'time_spend_company',

        3)#k, number of clusters
#PER SVIRGIO: SERVE PER PLOTTARE I CLUSTER E FORNIRE LA DESCRIZIONE DEI CLUSTER INDIVIDUATI CON VARI K [2-7]

dim='number_project'

df_lab = cstay.copy(deep=True)

df_lab['Labels'] = lab

hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

temp = pd.crosstab(cstay[dim], df_lab['Labels'])

#temp_g = temp.div(temp.sum(1).astype(float), axis=0)

temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

plt.xlabel(dim)

plt.ylabel('Cluster Label')

plt.show()
dim='last_evaluation'

#dim='satisfaction_level'

#dim='number_project'

#dim='average_montly_hours'

#dim='time_spend_company'

#dim='salary'

#dim='sales'

df_lab = dfb.loc[df['left'] == 1].copy(deep=True)

df_lab['Labels'] = lab

hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

temp = pd.crosstab(dfb[dim], df_lab['Labels'])

#temp_g = temp.div(temp.sum(1).astype(float), axis=0)

temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

plt.xlabel(dim)

plt.ylabel('Cluster Label')

plt.show()
others = cleft.copy(deep=True)

others['Labels'] = lab



others = others.loc[others['Labels'] ==0]

minS=min(others['satisfaction_level'])

maxS=max(others['satisfaction_level'])

meanS=mean(others['satisfaction_level'])

stdS=std(others['satisfaction_level'])

print(str(minS)+" - "+str(maxS))

print(str(meanS-stdS)+" - "+str(meanS+stdS))
min(dfb['average_montly_hours'])
clust3D(cstay,#dims to drop

        #'last_evaluation',

        'satisfaction_level',

        'number_project',

        #'average_montly_hours',

        'time_spend_company',

        10)#k, number of clusters
# single linkage and dendogram using scipy

dendo(df_clus,'euclidean','complete',0.78)
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
#Nuova funzione per DBSCAN ctay

#parametri che vanno bene anche per number_proj eps=0.166 e min_pts=145

#eps=0.1995 e minpts=285

#provati con eps 0.2 fino a 0.28 e minpt da 450 a 800

#potrebbe essere interessante 0.21, 336

#0.258, 790

par_eps=0.235#eps

par_minpts=500#minpts

#number_project

dim1='last_evaluation'

dim2='number_project'

dim3='average_montly_hours'

dbscan = DBSCAN(eps=par_eps, min_samples=par_minpts, metric='euclidean')

dbscan.fit(cstay)

ax = Axes3D(plt.figure(), rect=[0, 0, .95, 1], elev=48, azim=-134)

ax.set_xlim([0,1])

ax.set_ylim([0,1])

ax.set_zlim([0,1])

ax.set_xlabel(dim1)

ax.set_ylabel(dim2)

ax.set_zlabel(dim3)

lab=dbscan.labels_

#ax.set_title('silhouette:'+ str(silhouette_score(df_clus, lab)), fontsize=15, fontweight='bold')

ax.scatter(cstay[dim1],cstay[dim2],cstay[dim3],c=lab)



#0.17,160 diminuisce il rumore 3 cluster

#0.173, 160  2 cluster



dim='time_spend_company'

df_lab = cstay.copy(deep=True)

df_lab['Labels'] = lab









hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

temp = pd.crosstab(cstay[dim], df_lab['Labels'])

#temp_g = temp.div(temp.sum(1).astype(float), axis=0)

temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

plt.xlabel(dim)

plt.ylabel('Cluster Label')

plt.show()
# density based clustering

#0.210#eps

#480#minpts

#Prove:

#min_samples 750 eps 0.25 3 cluster di left

#min_samples 750 eps 0.251 3 cluster di left + W

#min_samples 200 eps 0.2

#min_samples 100 eps 0.17

#min_samples 75 eps 0.19

#min_samples 10 eps 0.1727

#min_samples 9 eps 0.1695

#min_samples 32 eps 0.42 high silhouette, but just 1 cluster

#min_samples 45 eps 0.07



dbscan3d(df_clus

         ,0.210#eps

         ,480#minpts

         ,'last_evaluation'

         ,'satisfaction_level'

         #'number_project'

         ,'average_montly_hours'

         #,'time_spend_company'

         )
others = df_clus.copy(deep=True)

others['Labels'] = lab

others = others.loc[others['Labels'] !=-1]#1left medium time_company, high last, low sat

print(len(others.loc[others['Labels'] ==2]))#1left medium time_company, high last, low sat

dfb['Labels']=lab

print(mean(dfb.loc[dfb['Labels'] ==2]))#1left medium time_company, high last, low sat

print ("DEVIAZIONE STANDARD")

print(std(dfb.loc[dfb['Labels'] ==2]))

dim='left'



#dim='last_evaluation'

#dim='number_project'

#dim='average_montly_hours'

#dim='time_spend_company'

#lab = [x for x in lab if x !=1 and x !=4 and x !=5]

hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

temp = pd.crosstab(df[dim], others['Labels'])

#temp_g = temp.div(temp.sum(1).astype(float), axis=0)

temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

plt.xlabel(dim)

plt.ylabel('Cluster Label')

plt.show()
#ANALISI del solo rumore

othersn=[]

if(first):

    others = df_clus.copy(deep=True)

    others['Labels'] = lab

    others = others.loc[others['Labels'] ==-1]#1left, low time_company, high last

    #othersn = dfb.copy(deep=True)

    #othersn['Labels'] = lab

    dfb = dfb.loc[dfb['Labels'] ==-1]#1left, low time_company, high last

    first = False

    

par_df=others

par_eps=0.192#eps

par_minpts=205 #minpts

dim1= 'last_evaluation'

dim2= 'satisfaction_level'

         #'number_project'

dim3= 'average_montly_hours'

dbscan = DBSCAN(eps=par_eps, min_samples=par_minpts, metric='euclidean')

dbscan.fit(par_df)

ax = Axes3D(plt.figure(), rect=[0, 0, .95, 1], elev=48, azim=-134)

ax.set_xlim([0,1])

ax.set_ylim([0,1])

ax.set_zlim([0,1])

ax.set_xlabel(dim1)

ax.set_ylabel(dim2)

ax.set_zlabel(dim3)

lab=dbscan.labels_

#ax.set_title('silhouette:'+ str(silhouette_score(df_clus, lab)), fontsize=15, fontweight='bold')

ax.scatter(others[dim1],others[dim2],others[dim3],c=lab)



#par_eps=0.192#eps, par_minpts=205#minpts iniziano a formarsi 3 clusters

#others = others.loc[others['Labels'] !=2]#1left medium time_company, high last, low sat

#dim= 'last_evaluation'

#dim= 'satisfaction_level'

#dim='number_project'

#dim= 'average_montly_hours'

dim='last_evaluation'

#df_lab = othersn.copy(deep=True)

#df_lab['Labels'] = lab

dfb['Labels']=lab

print(len(dfb.loc[dfb['Labels'] == 2]))

print(mean(dfb.loc[dfb['Labels'] == 2]))

print("")

print("DEVIAZIONE STANDARD")

print(std(dfb.loc[dfb['Labels'] == 2]))



hist, bins = np.histogram(lab, bins=range(0, len(set(lab)) + 1))

temp = pd.crosstab(df[dim], df_lab['Labels'])

#temp_g = temp.div(temp.sum(1).astype(float), axis=0)

temp.plot(kind='bar', stacked=True, title='Cluster Label Rate by '+dim)

plt.xlabel(dim)

plt.ylabel('Cluster Label')

plt.show()
#stima di min_samples

min_samples=50

df_temp=cstay

data_dist = squareform(pdist(df_temp, metric='euclidean'))

kth_neighbours = []

for i in range (0, len(data_dist)):

    kth_neighbours.append(np.partition(data_dist[i],min_samples)[min_samples])

plt.plot(range(0,len(df_temp)),sorted(kth_neighbours), 'k', linewidth=2)
#stima di eps

eps=0.09

df_temp=cstay

data_dist = squareform(pdist(df_temp, metric='euclidean'))

k_neighbours = []

for i in range (0, len(data_dist)):

    k_neighbours.append(len(numpy.where(data_dist[i]<eps)[0]))

plt.plot(range(0,len(df_temp)),sorted(k_neighbours), 'k', linewidth=2)
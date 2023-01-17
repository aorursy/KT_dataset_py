import numpy as np 

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.covariance import MinCovDet

import scipy as sc

import plotly.graph_objects as go

from scipy.spatial.distance import mahalanobis, euclidean

from scipy.stats import chi2

from scipy.stats import multivariate_normal

from scipy.linalg import sqrtm

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.neighbors import LocalOutlierFactor

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest

import scikitplot as skplt
data_raw=pd.read_csv("/kaggle/input/swiss-banknote-conterfeit-detection/banknotes.csv")
data_raw.head()
#Drop conterfeit information

data=data_raw.drop(["conterfeit"],axis=1).copy()
f, ax = plt.subplots(2,3,figsize=(18, 8))

columns=data.columns.values

for index,name in enumerate(columns):

    x,y=(0,index) if index<3 else (1,len(columns)-index-2)

    sns.distplot(data[name],ax=ax[x][y])
f, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(data.corr())

plt.show()
def get_mcd_outliers(data,threshold,type="outlier"):

    #MCD Estimators and Mahalanobis distances of data

    mcd_estimator=MinCovDet().fit(data)

    mahalanobis=mcd_estimator.mahalanobis(data)

    #Calculate outliers based on threshold

    if type=="outlier":

        transformation = lambda x: 0 if x <= threshold else 1

    elif type=="weights":

        transformation = lambda x: 1 if x <= threshold else 0

    outliers = np.array([transformation(xi) for xi in mahalanobis])

    return outliers



def get_fsrmcd_estimators(data,threshold,n,p,type="estimator"):

    weights=get_mcd_outliers(data,threshold,"weights")

    #FSRMD Location

    fsrmcd_location = np.matrix(np.average(data, axis=0, weights=weights))



    #FSRMCD Covariance

    fsrmcd_covariance = 0

    for x in range(0, n):

        fsrmcd_covariance = fsrmcd_covariance + weights[x] * np.dot(

            np.transpose((data.loc[x].values - fsrmcd_location)),

            (data.loc[x].values - fsrmcd_location))



    k_constant=(np.floor(0.975*n)/n)/(chi2.cdf(threshold,p+2))



    fsrmcd_covariance=(k_constant/(sum(weights)-1))*fsrmcd_covariance

    #Returns estimators

    if type=="estimator":

        return (fsrmcd_location,fsrmcd_covariance)

    

    #Returns outliers for FSRMCD method only

    elif type=="outlier":

        #Calculate mahalanobis distances and determine the outliers

        mahalanobis_fsrmcd=np.array([])

        for x in range(0,n):    

            maha=np.power(sc.spatial.distance.mahalanobis(data.loc[x].values,fsrmcd_location,np.linalg.inv(fsrmcd_covariance)),2)

            mahalanobis_fsrmcd=np.append(mahalanobis_fsrmcd,maha) 

        transformation = lambda x: 0 if x <= threshold else 1

        outliers = np.array([transformation(xi) for xi in mahalanobis_fsrmcd])

        return outliers





def get_fsrmcdmac_outliers(data):

    #Data shape

    n=len(data)

    p=data.shape[1]

    

    #Threshold Chi-Square Distribution

    threshold = chi2.ppf(0.975, p)

    

    #Get FSRMCD Estimators

    fsrmcd_location,fsrmcd_covariance=get_fsrmcd_estimators(data,threshold,n,p)



    #Calculate bandwidth matrix

    #Split matrix in pre factor hb and matrix H

    hb_factor=np.power((4/(4+p)),(1/(p+6)))*np.power(n,(-1/(p+6)))

    #Factor is different for different values for p, see the paper for details

    cnp_factor=0.7532+37.51*np.power(n,-0.9834)

    H=sqrtm(cnp_factor*fsrmcd_covariance)

    H_inverse=np.linalg.inv(H)



    #Copy and standardize data

    data_standardized = data.copy()

    for x in range(0, n):

        data_standardized.loc[x] = np.reshape(

            np.dot(H_inverse, np.transpose((data_standardized.loc[x].values - fsrmcd_location))),p)



    #Mode Association Clustering Algorithm

    modes=np.zeros((p,n))

    for x in range(0,n):



        #Select each datapoint as starting point from where to find the local maximum

        x0=data_standardized.loc[x].values

        x0_old=x0+1



        #Define stopping-criteria

        cnt=0

        err=100



        #Iterative algorithm to find the maximum

        while ((err>0.0001) and (cnt<150)):

            diag=np.zeros((p,p),int)

            np.fill_diagonal(diag,1)

            kde=multivariate_normal.pdf(data_standardized,x0,hb_factor*diag)



            d=kde/sum(kde)

            x0=np.dot(np.transpose(d),data_standardized)



            err=np.linalg.norm(x0-x0_old,ord=2)/np.maximum(np.linalg.norm(x0,ord=2),1)



            x0_old=x0

            cnt=cnt+1



        modes[:,x]=x0



    #Put different modes into different clusters

    clusters=np.zeros((1,n))

    clust_cnt=0



    err=1/(2*n)



    for x in range(0,n):

        if clusters[:,x]==0:

            clust_cnt=clust_cnt+1

            clusters[:,x]=clust_cnt



            for y in range(0,n):

                if clusters[:,y]==0:

                    if np.linalg.norm((modes[:,x]-modes[:,y]),ord=2)<err:

                        clusters[:,y]=clust_cnt



    #Get largest cluster and corresponding mode

    clust_max=-1

    clust_s=0

    for x in range(1,clust_cnt+1):

        s=len(clusters[clusters==x])



        if s>clust_s:

            clust_max=x

            clust_s=s



            ind=min(clusters[clusters==x])

            mode=modes[:,int(ind)]



    bulk=np.where(clusters==clust_max)[1]



    #Get mode by reverting the standardization

    mode=np.dot(H,mode)+fsrmcd_location



    #Save final Cluster

    if len(data.loc[bulk])<(p+1):

        cluster_final=data

    else:

        cluster_final=data.loc[bulk]

        cluster_final.reset_index(inplace=True)

        cluster_final.drop("index",axis=1,inplace=True)



    #Get estimators from cluster

    mean_cluster = cluster_final.mean().values

    covariance_cluster = cluster_final.cov().values

    weights=np.array([])



    #Get Mahalanobis distance and outliers

    for x in range(0, len(cluster_final)):

        maha = np.power(sc.spatial.distance.mahalanobis(cluster_final.loc[x].values, mean_cluster,

                                               np.linalg.inv(covariance_cluster)),2)

        if maha <= threshold:

            weights=np.append(weights,1)

        else:

            weights=np.append(weights,0)





    #Get final robust estimators

    #Location

    robust_location = np.matrix(np.average(cluster_final, axis=0, weights=weights))



    #Covariance

    robust_covariance = 0

    for x in range(0, len(cluster_final)):

        robust_covariance = robust_covariance + weights[x] * np.dot(

            np.transpose((data.loc[x].values - robust_location)),

            (data.loc[x].values - robust_location))



    robust_covariance=(1/(sum(weights)-1))*robust_covariance



    #Calculate final mahalanobis distances and determine the final outliers

    mahalanobis_robust=np.array([])

    for x in range(0,n):    

        maha=np.power(sc.spatial.distance.mahalanobis(data.loc[x].values,robust_location,np.linalg.inv(robust_covariance)),2)

        mahalanobis_robust=np.append(mahalanobis_robust,maha)    



    #Outlier thresholds

    #L1 and L2 depend on the dimension of the dataset. See the paper for values for different dimensions

    L1=31.9250

    L2=16.9710

    if mahalanobis_robust.max()<L1:

        outliers=np.repeat(0,200)

    else:

        outliers=np.array([])

        for x in range(0,n):

            if mahalanobis_robust[x]>L2:

                outliers=np.append(outliers,1)

            else:

                outliers=np.append(outliers,0)

    return outliers
results=data_raw.copy()

results["Outlier_FSRMCDMAC"]=get_fsrmcdmac_outliers(data)

results["Outlier_FSRMCDMAC"]=results["Outlier_FSRMCDMAC"].astype(int)

print("Accuracy: "+str(accuracy_score(results["conterfeit"],results["Outlier_FSRMCDMAC"])))

skplt.metrics.plot_confusion_matrix(results["conterfeit"],results["Outlier_FSRMCDMAC"],normalize="true")

plt.show()
#Isolation Forest

isf=IsolationForest()

isf_outliers=isf.fit_predict(data)

results["Outlier_ISF"]=isf_outliers

results["Outlier_ISF"]=results["Outlier_ISF"].map({1:0,-1:1})



#Local Outlier Factor

lof=make_pipeline(StandardScaler(),LocalOutlierFactor())

lof_outliers=lof.fit_predict(data)

results["Outlier_LOF"]=lof_outliers

results["Outlier_LOF"]=results["Outlier_LOF"].map({1:0,-1:1})



#Mahalanobis Distance with MCD Estimators

p=data.shape[1]

threshold = chi2.ppf(0.975, p)

mcd_outliers = get_mcd_outliers(data,threshold)

results["Outlier_MCD"]=mcd_outliers



#Compare different methods

methods=["Outlier_ISF","Outlier_LOF","Outlier_MCD","Outlier_FSRMCDMAC"]

comparison=pd.DataFrame(columns=["TN","FP","FN","TP"])

for method in methods:

    comparison.loc[method]=confusion_matrix(results["conterfeit"],results[method]).reshape(4)

comparison["Accuracy"]=(comparison["TP"]+comparison["TN"])/len(data)

comparison["Sensitivity"]=(comparison["TP"])/(len(data)/2)

comparison["Specificity"]=(comparison["TN"])/(len(data)/2)

comparison.sort_values(by="Sensitivity",inplace=True)

comparison
methods=["Local Outlier Factor","MCD","Isolation Forest","FSRMCD-MAC"]

fig=go.Figure()

fig.add_trace(go.Bar(y=methods,x=comparison["TN"]/100,orientation="h",name="Correctly identified genuine banknotes (TNR) (%)"))

fig.add_trace(go.Bar(y=methods,x=comparison["TP"]/100,orientation="h",name="Correctly identified fake banknotes (TPR) (%)"))
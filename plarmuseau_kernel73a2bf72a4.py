import pandas as pd

import numpy as np

import os

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import geopandas as gpd

import pycountry

os.popen('cd ../input/big-five-personality-test/IPIP-FFM-data-8Nov2018; ls').read()

path = r'../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv'

df_full = pd.read_csv(path, sep='\t')

pd.options.display.max_columns = 999

df_full.head()

df_full=df_full[df_full.IPC==1]
def kluster(data,grbvar,label,nummercl,level):

    '''nummercl < ncol'''





    from sklearn.cluster import KMeans

    from sklearn.metrics.pairwise import cosine_similarity

    from sklearn.metrics import confusion_matrix

    import matplotlib.pyplot as plt

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis

    from sklearn.decomposition import PCA,TruncatedSVD,NMF,FastICA

    from umap import UMAP  # knn lookalike of tSNE but faster, so scales up

    from sklearn.manifold import TSNE,Isomap,SpectralEmbedding,spectral_embedding,LocallyLinearEmbedding,MDS #limit number of records to 100000

    ytrain=data[label]

    if True:

        from category_encoders.cat_boost import CatBoostEncoder

        CBE_encoder = CatBoostEncoder()

        cols=[ci for ci in data.columns if ci not in ['index',label]]

        coltype=data.dtypes

        featured=[ci for ci in cols]

        ytrain=data[label]

        data = CBE_encoder.fit_transform(data.drop(label,axis=1), ytrain)

        data[label]=ytrain



    clusters = [PCA(n_components=nummercl,random_state=0,whiten=True),

                TruncatedSVD(n_components=nummercl, n_iter=7, random_state=42),

                FastICA(n_components=nummercl,random_state=0),

                NMF(n_components=nummercl,random_state=0),

                Isomap(n_components=nummercl),

                LocallyLinearEmbedding(n_components=nummercl),

                #SpectralEmbedding(n_components=nummercl),

                #MDS(n_components=nummercl),

                TSNE(n_components=3,random_state=0),

                UMAP(n_neighbors=nummercl,n_components=10, min_dist=0.3,metric='minkowski'),

                ] 

    clunaam=['PCA','tSVD','ICA','NMF','Iso','LLE','Spectr','MDS','tSNE','UMAP']

    

    grbdata=data.groupby(grbvar).mean()

    simdata = cosine_similarity(grbdata.fillna(0))

    if len(grbdata)<3:

        simdata=data#.drop(grbvar,axis=1)

        simdata=simdata.dot(simdata.T)

        from sklearn import preprocessing

        simdata = preprocessing.MinMaxScaler().fit_transform(simdata)



    for cli in clusters:

        print(cli)

        clunm=clunaam[clusters.index(cli)] #find naam

        if clunm=='NMF':

            simdata=simdata-simdata.min()+1

        svddata = cli.fit_transform(simdata)



        km = KMeans(n_clusters=nummercl, random_state=0)

        km.fit_transform(svddata)

        cluster_labels = km.labels_

        clulabel='Clu'+clunm+str(level)

        cluster_labels = pd.DataFrame(cluster_labels, columns=[clulabel])

        print(cluster_labels.head())

        pd.DataFrame(svddata).plot.scatter(x=0,y=1,c=cluster_labels[clulabel].values,colormap='viridis')

        print(clunm,cluster_labels.mean())

        plt.show()



        clusdata=pd.concat([pd.DataFrame(grbdata.reset_index()[grbvar]), cluster_labels], axis=1)

        if len(grbdata)<3: 

            data['Clu'+clunm+str(level)]=cluster_labels.values

            

        else:

            data=data.merge(clusdata,how='left',left_on=grbvar,right_on=grbvar)

        

        print('Correlation\n',confusion_matrix ( data[label],data[clulabel]))

            

    return data

train2=kluster(df_full.iloc[:,:].sample(10000).fillna(0).reset_index(),'country','index',5,1)

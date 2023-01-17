# load in our libraries

import pandas as pd # pandas for data frames

from scipy.stats import ttest_ind # just the t-test from scipy.stats

from scipy.stats import probplot # for a qqplot

import matplotlib.pyplot as plt # for a qqplot

import pylab #



# read in our data

cereals = pd.read_csv("../input/cereal.csv")

# check out the first few lines

cereals
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

                SpectralEmbedding(n_components=nummercl),

                MDS(n_components=nummercl),

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

total=kluster(cereals.drop('name',axis=1),'mfr','fat',6,1)
# plot a qqplot to check normality. If the varaible is normally distributed, most of the points 

# should be along the center diagonal.

probplot(cereals["sodium"], dist="norm", plot=pylab)
# get the sodium for hot cerals

hotCereals = cereals["sodium"][cereals["type"] == "H"]

# get the sodium for cold ceareals

coldCereals = cereals["sodium"][cereals["type"] == "C"]



# compare them

ttest_ind(hotCereals, coldCereals, equal_var=False)
# let's look at the means (averages) of each group to see which is larger

print("Mean sodium for the hot cereals:")

print(hotCereals.mean())



print("Mean sodium for the cold cereals:")

print(coldCereals.mean())
# plot the cold cereals

plt.hist(coldCereals, alpha=0.5, label='cold')

# and the hot cereals

plt.hist(hotCereals, label='hot')

# and add a legend

plt.legend(loc='upper right')

# add a title

plt.title("Sodium(mg) content of cereals by type")
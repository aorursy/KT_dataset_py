import numpy as np

import pandas as pd

import matplotlib as plt

from matplotlib import pyplot



# Read fpkm file

fpkm=open('/kaggle/input/chol-fpkm/TCGA-CHOL.htseq_fpkm.tsv', 'r')

fpkmlist=fpkm.readlines()



genelabels=[]



# Parse the file

for i in range(len(fpkmlist)):

    fpkmlist[i]=fpkmlist[i].split('\t')

    

    # Get rid of line endings

    fpkmlist[i][len(fpkmlist[i])-1]=fpkmlist[i][len(fpkmlist[i])-1][:len(fpkmlist[i][len(fpkmlist[i])-1])-1]

    

    genelabels.append(fpkmlist[i][0])

    

    # Keep just the fpkm values

    fpkmlist[i]=fpkmlist[i][1:]



samplelabels=fpkmlist[0]

    

fpkmlist=fpkmlist[1:]



genelabels=genelabels[1:]





# Centering the data

for i in range(len(fpkmlist)):

    fpkmlist[i]=list(map(float, fpkmlist[i]))

    temp=sum(fpkmlist[i])/len(fpkmlist[i])

    for j in range(len(fpkmlist[i])):

        fpkmlist[i][j]-=temp



# Turing it into a dataframe

df=pd.DataFrame(fpkmlist)



# Turning the dataframe into a suitable format for PCA - rows being samples and columns being genes

df=df.T



# Making indexing more convenient

df.columns=genelabels

samplelabels=dict(zip([i for i in range(45)], samplelabels))

df=df.rename(index=samplelabels)





# I wanted to try doing some analysis without a library first



dfcov=np.cov(df)



eigenvalues, eigenvectors=np.linalg.eig(dfcov)



eigenvectors=eigenvectors.transpose()



c = list(zip(eigenvalues, eigenvectors))

c.sort()

c.reverse()

eigenvalues, eigenvectors = zip(*c)



print('1st component EVR=',eigenvalues[0]/sum(eigenvalues))

print('2nd component EVR=',eigenvalues[1]/sum(eigenvalues))



# So here we can see that after the first component the explained variance ratio (how much of the original info on the data is left) drops drastically, so most likely it is the only useful one



# Did it with a library just to make sure



from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(df)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])



principalDf.plot.scatter('principal component 1', 'principal component 2');



# The plot shows a pretty clear cluster of nine samples that lie to the right of the 100 mark on the axis of the first principal component



# Removing the samples selected by PCA from our dataset

droplist=[]

for i in range(45):

    if principalDf.at[i,'principal component 1']>100:

        droplist.append(samplelabels[i])



df.drop(droplist, inplace=True)



print('Samples to get rid of:', droplist)

                  

# Assigning a number to each sample for future reference

samplelabels=dict(zip([i for i in range(len(list(df.index.values)))], list(df.index.values)))

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics

from sklearn.metrics import pairwise_distances





npdf=df.to_numpy()



# Trying two clustering methods:

# K-means, which creates and adjusts centroids, trying to minimize the inertia (the overall difference between a cluster's centroid and the elements of that cluster)

# And another method which is similar and also tries to minimize the sum of differences within all clusters, but does it with a hierarchical approach, when each point starts in its own cluster and then pairs of clusters that are closest become a higher-order cluster





# So first lets see if there is a point (number of clusters) where the inertia levels out

# And evaluate the clustering for each nuber of possible clusters with the silhouette score, which basically compares the mean distance between a point and all other points in the same class with the mean distance between a sample and all other points in the next nearest cluster



inertias = []

kscores=[]

for i in range(2, 36):

    kmeans = KMeans(n_clusters=i)

    kmeans.fit(npdf)

    inertias.append(kmeans.inertia_)

    kscores.append(metrics.silhouette_score(npdf, kmeans.labels_, metric='euclidean'))



print('K-means silhouette scores for different numbers of clusters:')

for i in range(34):

      print(i+2, 'clusters, score=', kscores[i])

    

plt.plot(range(2, 36), inertias)

plt.xlabel('Number of clusters')

plt.ylabel('Inertia')

plt.show()



# We can see there is no number of clusters at which the inertia levels out, so we'll take the number where the silhouette score is highest





    


hscores=[]

for i in range(2, 36):

    hierarchical = AgglomerativeClustering(n_clusters=i).fit(npdf)

    hscores.append(metrics.silhouette_score(npdf, hierarchical.labels_, metric='euclidean'))

    

print('Hierarchical silhouette scores for different numbers of clusters:')

for i in range(34):

      print(i+2, 'clusters, score=', hscores[i])



kmeans = KMeans(n_clusters=kscores.index(max(kscores))+2)

kmeans.fit(npdf)



hierarchical = AgglomerativeClustering(n_clusters=hscores.index(max(hscores))+2).fit(npdf)





# Let's go with the method that had the highest silhouette score



# Separating sample names into the two clusters

cluster1=[]

cluster2=[]



if max(kscores)>max(hscores):



    for i in range(len(kmeans.labels_)):

        if kmeans.labels_[i]==1:

            cluster1.append(samplelabels[i])

        else:

            cluster2.append(samplelabels[i])

            

else:

    

    for i in range(len(hierarchical.labels_)):

        if hierarchical.labels_[i]==1:

            cluster1.append(samplelabels[i])

        else:

            cluster2.append(samplelabels[i])

    
import numpy as np

import pandas as pd



muse=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.muse_snv.tsv', sep='\t')

mutect2=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.mutect2_snv.tsv', sep='\t')

somaticsniper=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.somaticsniper_snv.tsv', sep='\t')

varscan2=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.varscan2_snv.tsv', sep='\t')



muse=muse.to_numpy()

mutect2=mutect2.to_numpy()

somaticsniper=somaticsniper.to_numpy()

varscan2=varscan2.to_numpy()



musemutect2=[]



somaticsnipervarscan2=[]



allofthem=[]



# Find mutations that are present in all datasets generated by the four programs



for i in range(len(muse)):

    for j in range(len(mutect2)):

        if list(muse[i])==list(mutect2[j]):

            musemutect2.append(muse[i])

            

for i in range(len(somaticsniper)):

    for j in range(len(varscan2)):

        if list(somaticsniper[i])==list(varscan2[j]):

            somaticsnipervarscan2.append(somaticsniper[i])

            

for i in range(len(musemutect2)):

    for j in range(len(somaticsnipervarscan2)):

        if list(musemutect2[i])==list(somaticsnipervarscan2[j]):

            allofthem.append(musemutect2[i])

            

print('Number of common mutations=', len(allofthem))
import numpy as np

import pandas as pd



oncokb=pd.read_csv('/kaggle/input/oncokb/cancerGeneList.tsv', sep='\t')

oncokb=oncokb.set_index('Hugo Symbol') 



mutgenes=open('/kaggle/input/oncokb/Mutated_Genes.txt', 'r')

mutgeneslist=mutgenes.readlines()



cutoffmut=[]



# Filtering out genes with mutations frequencies below 10%

for i in range(1, len(mutgeneslist)):

    mutgeneslist[i]=mutgeneslist[i].split('	')

    if float(mutgeneslist[i][len(mutgeneslist[i])-2][:len(mutgeneslist[i][len(mutgeneslist[i])-2])-1])>10:

        cutoffmut.append(mutgeneslist[i][0])

        

oncogenes=[]

oncosupressors=[]



# Selecting oncogenes and oncosupressors from the filtered genes

for i in range(len(cutoffmut)):

    if cutoffmut[i] in oncokb.index:

        if (oncokb.at[cutoffmut[i],'Is Oncogene']=='Yes'):

            oncogenes.append(cutoffmut[i])

        if (oncokb.at[cutoffmut[i],'Is Tumor Suppressor Gene']=='Yes'):

            oncosupressors.append(cutoffmut[i])

            

print('Oncogenes:', oncogenes)

print('Oncosupressors:', oncosupressors)
import numpy as np

import pandas as pd



muse=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.muse_snv.tsv', sep='\t')

mutect2=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.mutect2_snv.tsv', sep='\t')

somaticsniper=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.somaticsniper_snv.tsv', sep='\t')

varscan2=pd.read_csv('/kaggle/input/mutations/TCGA-CHOL.varscan2_snv.tsv', sep='\t')



selectedgenes=oncogenes+oncosupressors



# Checking to see which samples have mutations in the selected oncogenes and oncosupressors



mutsampledict={}



for i in list(muse.index.values):

    if muse.at[i,'gene'] in selectedgenes:

        if muse.at[i,'gene'] in mutsampledict.keys():

            mutsampledict[muse.at[i,'gene']].append(muse.at[i,'Sample_ID'])

        else:

            mutsampledict[muse.at[i,'gene']]=[muse.at[i,'Sample_ID']]

            

for i in list(mutect2.index.values):

    if mutect2.at[i,'gene'] in selectedgenes:

        if mutect2.at[i,'gene'] in mutsampledict.keys():

            mutsampledict[mutect2.at[i,'gene']].append(mutect2.at[i,'Sample_ID'])

        else:

            mutsampledict[mutect2.at[i,'gene']]=[mutect2.at[i,'Sample_ID']]

            

for i in list(somaticsniper.index.values):

    if somaticsniper.at[i,'gene'] in selectedgenes:

        if somaticsniper.at[i,'gene'] in mutsampledict.keys():

            mutsampledict[somaticsniper.at[i,'gene']].append(somaticsniper.at[i,'Sample_ID'])

        else:

            mutsampledict[somaticsniper.at[i,'gene']]=[somaticsniper.at[i,'Sample_ID']]

            

for i in list(varscan2.index.values):

    if varscan2.at[i,'gene'] in selectedgenes:

        if varscan2.at[i,'gene'] in mutsampledict.keys():

            mutsampledict[varscan2.at[i,'gene']].append(varscan2.at[i,'Sample_ID'])

        else:

            mutsampledict[varscan2.at[i,'gene']]=[varscan2.at[i,'Sample_ID']]



for i in mutsampledict.keys():

    print(i+':', mutsampledict[i])

    print('\n')
geneclustercount={}



# Seeing how many samples with mutations in the selected oncogenes and oncosupressors belong to each cluster



for i in mutsampledict.keys():

    geneclustercount[i]=['1st cluster:', len(list(set(mutsampledict[i]).intersection(cluster1))), '2nd cluster:', len(list(set(mutsampledict[i]).intersection(cluster2)))]





differencelist=[]

for i in geneclustercount.keys():

    differencelist.append(abs(geneclustercount[i][1]-geneclustercount[i][3]))



genes=list(geneclustercount.keys())

    

c = list(zip(differencelist, genes))

c.sort()

c.reverse()

differencelist, genes = zip(*c)



geneclusterdifference=dict(zip(genes, differencelist))



# Seeing the difference between the clusters, the most significant differences being presented first



print('The difference in mutation presence for the selected genes between the two clusters:')

print(geneclusterdifference)

     
# Installing the neccessary packages



!pip install lifelines



from lifelines import KaplanMeierFitter



# Extracting the needed data



survival=pd.read_csv('/kaggle/input/survival/TCGA-CHOL.survival.tsv', sep='\t')

survival=survival.to_numpy()



durations1 = []

event_observed1 = []

durations2 = []

event_observed2 = []



for i in range(1, len(survival)):

    if survival[i][0] in cluster1:

        durations1.append(survival[i][3])

        event_observed1.append(survival[i][1])

    if survival[i][0] in cluster2:

        durations2.append(survival[i][3])

        event_observed2.append(survival[i][1])



# Plotting Kaplan-Meier Curves (the probability of surviving in a given length of time) for the two clusters

        

kmf = KaplanMeierFitter() 



kmf.fit(durations1, event_observed1,label='Kaplan Meier Estimate for 1st cluster')



kmf.plot(ci_show=False);



kmf.fit(durations2, event_observed2,label='Kaplan Meier Estimate for 2nd cluster')



kmf.plot(ci_show=False);



# So here we can see that survival probability at a given moment in time is overall higher for the second cluster



# The survival probability is calculated as the number of subjects surviving at a given moment divided by the number of subjects who were at risk of dying at a given moment
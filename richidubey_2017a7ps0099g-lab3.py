# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv");
df.dtypes
df.head(10)

#from sklearn.preprocessing import LabelEncoder

#gle = LabelEncoder()

#gender_label = gle.fit_transform(df['gender'])

#gender_mappings = {index: label for index, label in 

 #                 enumerate(gle.classes_)}

#gender_mappings



from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

df.iloc[:, 1] = labelencoder.fit_transform(df.iloc[:, 1])







df.head(10)

#df['gender']=gender_mappings

df.iloc[:, 3] = labelencoder.fit_transform(df.iloc[:, 3])

df.iloc[:, 4] = labelencoder.fit_transform(df.iloc[:, 4])

df.iloc[:, 5] = labelencoder.fit_transform(df.iloc[:, 5])

df.iloc[:, 6] = labelencoder.fit_transform(df.iloc[:, 6])

df.iloc[:, 7] = labelencoder.fit_transform(df.iloc[:, 7])

df.iloc[:, 8] = labelencoder.fit_transform(df.iloc[:, 8])

df.iloc[:, 9] = labelencoder.fit_transform(df.iloc[:, 9])

df.iloc[:, 10] = labelencoder.fit_transform(df.iloc[:, 10])

df.iloc[:, 11] = labelencoder.fit_transform(df.iloc[:, 11])

df.iloc[:, 12] = labelencoder.fit_transform(df.iloc[:, 12])

df.iloc[:, 13] = labelencoder.fit_transform(df.iloc[:, 13])

df.iloc[:, 14] = labelencoder.fit_transform(df.iloc[:, 14])

df.iloc[:, 15] = labelencoder.fit_transform(df.iloc[:, 15])

df.iloc[:, 17] = labelencoder.fit_transform(df.iloc[:, 17])

#df['TotalCharges'] =(float)df['TotalCharges']

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.replace(' ',''))



df.head(10)
df.dtypes



df.corr()
import seaborn as sns

sns.heatmap(df.corr())
imp=["TVConnection","Channel3","Channel4","Channel5","Channel6","AddedServices","Subscription","tenure","PaymentMethod","MonthlyCharges","TotalCharges"]

#from sklearn.preprocessing import MinMaxScaler

#mms = MinMaxScaler()

#mms.fit(df)

#data_transformed = mms.transform(df)



df.fillna(df.mean(), inplace=True)



df[imp]
missing_count = df.isnull().sum()

missing_count[missing_count > 0]



# Implies No Null Value Present! Yay



#drop total charges as it has null values.!!
missing_count = df.isnull().sum()

missing_count[missing_count > 0]

from sklearn.cluster import KMeans



X = np.array(df[imp])



kmeans = KMeans(n_clusters=2, random_state=2).fit(X)
import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np

from sklearn.cluster import KMeans

plt.scatter(X[:,0],X[:,1], label='True Position')

print(kmeans.cluster_centers_)
print(kmeans.labels_)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
#check=np.concatenate(kmeans.labels_,df["Satisfied"])

#check



#for i in range(len(df)):

    #print("Predicted: ", kmeans.labels_[i] , " Real : ", df["Satisfied"][i])

    

    

TP=0;

TN=0;

FP=0;

FN=0;



for i in range(len(df)):

    predicted=kmeans.labels_[i];

    actual=df["Satisfied"][i];

    

    if(predicted==actual):

        if(actual==1):

            TP=TP+1;

        if(actual==0):

            TN=TN+1;

    else:

        if(actual==1 and predicted==0):

            FN=FN+1;

        else:

            FP=FP+1



            

print("PRoduced Output"," TP = ",TP,"TN=",TN,"FP=",FP,"FN=",FN)

    
correct = 0

for i in range(len(df)):

    if kmeans.labels_[i] == df['Satisfied'][i]:

        correct += 1



print(correct/len(X))



# Lets Tweak It Further Baby!

kmeans = KMeans(n_clusters=2, max_iter=2000, algorithm = 'auto')

kmeans.fit(df[imp])



from sklearn.cluster import MeanShift



clustering = MeanShift(bandwidth=2).fit(df[imp])



from sklearn import datasets, cluster

#digits = datasets.load_digits()



#images = digits.images

#X = np.reshape(images, (len(images), -1))

#agglo = cluster.FeatureAgglomeration(n_clusters=2)

#agglo.fit(df[imp]) 



from sklearn.cluster import AgglomerativeClustering



clustering = AgglomerativeClustering().fit(df[imp])



correct = 0

for i in range(len(df)):

    if clustering.labels_[i] == df['Satisfied'][i]:

        correct += 1



print(correct/len(X))



dfn=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')





from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

dfn.iloc[:, 1] = labelencoder.fit_transform(dfn.iloc[:, 1])

dfn.iloc[:, 3] = labelencoder.fit_transform(dfn.iloc[:, 3])

dfn.iloc[:, 4] = labelencoder.fit_transform(dfn.iloc[:, 4])

dfn.iloc[:, 5] = labelencoder.fit_transform(dfn.iloc[:, 5])

dfn.iloc[:, 6] = labelencoder.fit_transform(dfn.iloc[:, 6])

dfn.iloc[:, 7] = labelencoder.fit_transform(dfn.iloc[:, 7])

dfn.iloc[:, 8] = labelencoder.fit_transform(dfn.iloc[:, 8])

dfn.iloc[:, 9] = labelencoder.fit_transform(dfn.iloc[:, 9])

dfn.iloc[:, 10] = labelencoder.fit_transform(dfn.iloc[:, 10])

dfn.iloc[:, 11] = labelencoder.fit_transform(dfn.iloc[:, 11])

dfn.iloc[:, 12] = labelencoder.fit_transform(dfn.iloc[:, 12])

dfn.iloc[:, 13] = labelencoder.fit_transform(dfn.iloc[:, 13])

dfn.iloc[:, 14] = labelencoder.fit_transform(dfn.iloc[:, 14])

dfn.iloc[:, 15] = labelencoder.fit_transform(dfn.iloc[:, 15])

dfn.iloc[:, 17] = labelencoder.fit_transform(dfn.iloc[:, 17])

#df['TotalCharges'] =(float)df['TotalCharges']

dfn['TotalCharges'] = pd.to_numeric(dfn['TotalCharges'].str.replace(' ',''))

#imp=["TVConnection","Channel4","Channel5","Channel6","AddedServices","Subscription","tenure","PaymentMethod","MonthlyCharges"]

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()

mms.fit(dfn)

data_transformed = mms.transform(dfn)





dfn[imp]
dfn.fillna(dfn.mean(), inplace=True)



dfn.isnull().sum()
dfn.isnull().sum()


kmeansans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')

kmeansans.fit(dfn[imp])



clusteringans = AgglomerativeClustering().fit(dfn[imp])


finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':kmeansans.labels_})



finalans.to_csv('submissionkmeanimp.csv',index=False)

print("Lalla")





finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':clusteringans.labels_})



finalans.to_csv('submissionagloimp.csv',index=False)

print("Lalla2")



from sklearn.cluster import AgglomerativeClustering



cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

cluster.fit_predict(dfn[imp])



finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':cluster.labels_})



finalans.to_csv('submissionagglover2imp.csv',index=False)

print("Lalla3")



from sklearn.cluster import AgglomerativeClustering



cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')

cluster.fit_predict(dfn[imp])



finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':cluster.labels_})



finalans.to_csv('AggloAll.csv',index=False)

print("Lallastupid1")





kmeansans = KMeans(n_clusters=2, max_iter=600, algorithm = 'auto')

kmeansans.fit(dfn)



finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':kmeansans.labels_})



finalans.to_csv('KMeansAll.csv',index=False)

print("Lallastu[id2")



#from sklearn.cluster import SpectralClustering



#clustering = SpectralClustering(n_clusters=2,assign_labels="discretize",random_state=0).fit(dfn[imp])
#finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':clustering.labels_})



#finalans.to_csv('submissionspectral.csv',index=False)

#print("Spec done[id2")

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)

savegmm=gmm.fit_predict(dfn[imp])



savegmm
finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':savegmm})



finalans.to_csv('GMMimp.csv',index=False)

print("Lallastu[id2")

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2)

savegmm=gmm.fit_predict(dfn)

finalans=pd.DataFrame({'custId':dfn["custId"],'Satisfied':savegmm})



finalans.to_csv('GMMall.csv',index=False)

print("Lallastu[id2")





#Importing all the important libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 10)

from sklearn.cluster import KMeans

from sklearn.cluster import DBSCAN

from sklearn.decomposition import PCA

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from sklearn.cluster import Birch

from sklearn.cluster import AgglomerativeClustering
#from google.colab import files

#uploaded=files.upload()
#Reading the data

df=pd.read_csv("../input/dmassign1/data.csv")

data_orig=pd.read_csv("../input/dmassign1/data.csv")

#Id is randomly generated so we can remove it

df.drop(['ID'],axis=1,inplace=True)
df.info()
#replacing ? with NaN for smooth pre processing

df.replace('?', np.NaN, inplace = True)
df
#checking which columns have how many unique values to distinguish between categorical and numerical features

df_dtype_nunique = pd.concat([df.dtypes, df.nunique()],axis=1)

df_dtype_nunique.columns = ["dtype","unique"]

df_dtype_nunique
df.Col197.unique()

df.Col197.replace({"sm":"SM","me":"ME","la":"LA","M.E.":"ME"},inplace=True)
df.Col197.unique()
#There are multiple columns where certain blocks have numbers represented as a string,

#we need to convert such blocks to an appropriate float format to make it useful for further evaluation of data.

print(df.dtypes)
#covert object type to float type and save that column as categorical feature

columns=list(df)

numerical_features=[]



for i in columns:

  df[i]=df[i].astype(float)

  numerical_features.append(i)

  

#THE ERROR MESSAGE IS INTENDED AND NOT AN ERROR
numerical_features
#filling the Nan with mean

df.fillna(df.mean(), inplace=True)
df.info()
#checking if there are still missing values left

missing_count = df.isnull().sum()

missing_count[missing_count > 0]

missing_count
categorical_features=['Col189','Col190','Col191','Col192','Col193','Col194','Col195','Col196','Col197']
#storing the class labels in a different list for coparison with predicted data

class_label=df["Class"]
df["Class"]=df["Class"].astype(int)
#function to map predicted cluster to class labels

def mapping_from_cluster_to_class(cluster_labels, class_labels):

  mapping_cluster_to_class={}

  temporary_dict={}

  for cluster in np.unique(cluster_labels):

    mapping_cluster_to_class[cluster]=0

    for class_label in np.unique(class_labels):

      temporary_dict[(cluster,class_label)]=0

  

  for cluster in np.unique(cluster_labels):

    for i in range(len(class_labels)):

      if cluster_labels[i]==cluster:

        temporary_dict[(cluster,class_labels[i])]+=1



  for cluster in np.unique(cluster_labels):

    mapping_cluster_to_class[cluster]=max(np.unique(class_labels),key=lambda x:temporary_dict[(cluster, x)])



  return mapping_cluster_to_class
#getting dummy variables for categorical data

df=pd.get_dummies(data=df,columns=categorical_features)
#Creating the data set for training the model

X=df.loc[:,df.columns != 'Class']

y=class_label
y=y.astype(int)
#Standard scaling the whole dataset 

X.loc[:,"Col1":"Col197_XL"]=StandardScaler().fit_transform(X.loc[:,"Col1":"Col197_XL"])
#applying pca to reduce the dimension 

X=PCA(n_components=10).fit_transform(X)

#fitting various model and predicting the clusters

cluster_label_birch=Birch( n_clusters=20).fit(X).predict(X)

arg1_b=cluster_label_birch[0:1299]

arg2_b=y[0:1299]

mapping_b= mapping_from_cluster_to_class(arg1_b,arg2_b)

mapping_b
final_mapping_b=[]

for i in cluster_label_birch:

  final_mapping_b.append(mapping_b[i])



def accuracy(mapping):

  accurate=0

  for i in range(0,1300):

    if mapping[i]==y[i]:

      accurate += 1

  return accurate/1300


accuracy_b=accuracy(final_mapping_b)

accuracy_b
res2 = pd.DataFrame(final_mapping_b)

final_b = pd.concat([data_orig["ID"], res2], axis=1).reindex()

final_b = final_b.rename(columns={0: "Class"})

final_b=final_b.drop(df.index[0:1300])

final_b.head()

final_b.to_csv('mysub_b.csv',index=False)



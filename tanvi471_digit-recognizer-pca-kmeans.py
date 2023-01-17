import numpy as np 
import pandas as pd 
import pandas_profiling as pp
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

train = pd.read_csv("../input/train.csv")
train.head(5)
#pp.ProfileReport(train)
# Separating out the features
x = train.iloc[:,1:].values

# Separating out the label
y = train.loc[:,['label']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf2 = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf2, train[['label']]], axis = 1)
finalDf.head(5)
df_list = [x[1] for x in finalDf.groupby(('label'),sort=True)]

for i in range(1,len(df_list)):
    df_list[i]= df_list[i].reset_index()
    df_list[i] =df_list[i].drop(['index'], axis=1)
    
for i in range(0,10):
    df_list[i].plot(x="principal component 1",y="principal component 2", kind="scatter",label='%s data' %i)
x = train.iloc[:,1:].values

# Separating out the target
y = train.loc[:,['label']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)


pca = PCA(n_components=30)
principalComponents = pca.fit_transform(x)
principalDf30 = pd.DataFrame(data = principalComponents)

DIGITS_PCA_30 = pd.concat([principalDf30, train[['label']]], axis = 1)
DIGITS_PCA_30.head(5)
#Subsetting the data into clusters for 0-9, labelled data
df_list_30_label = [x[1] for x in DIGITS_PCA_30.groupby(('label'),sort=True)]

##Creating a list of dataframes and then utilizing single element(dataset) to sample 10 values 

for i in range(0,len(df_list_30_label)):
    df_list_30_label[i]= df_list_30_label[i].reset_index()
    df_list_30_label[i] =df_list_30_label[i].drop(['index'], axis=1)
    
##Creating empty lists so that elements can be appended later    
center_list=[]
km_iterlist=[]
purity=[]
class_label=[]
for i in range(0,10):
    class_label.append(i)
    #print(class_label)
    center=df_list_30_label[i].sample(n=10, replace=True)
    center_list.append(center)
    center_list[i]=center_list[i].drop(['label'], axis=1)
    
###initialize the same 10 values as the "init" parameter in Kmeans
    
    #print(center_list[i])
    ##The stopping criteria for each cluster convergence is dependent on the iterations which are 300 by default
    km = KMeans(n_clusters=10,init=center_list[i], n_init=1).fit(principalDf30)
    ykmeans = km.predict(principalDf30)
    df_temp = pd.DataFrame(y,ykmeans, columns=["True"]).reset_index()
    df_temp.head(5)
    majority_count= 0
    for i in df_temp["index"].unique():
        df_inner = df_temp[df_temp["index"] == i]
        count_true_values = df_inner["True"].value_counts()
        #print(count_true_values)
        majority_count += count_true_values.iloc[0]
    majority_purity = majority_count/42000
    purity.append(majority_purity)  
    #print(purity)
     
    km_iterlist.append(km.n_iter_)
    #print(km_iterlist)
    
Purity_ = pd.DataFrame()
Purity_["Class_label"] = class_label
Purity_["No.of Iteration"] = km_iterlist
Purity_["Purity factor"] = purity

Purity_
###Initialize the 10 cluster centers – one from each class
### The only difference in the code is the init parameter in the Kmeans algorithm , wherein the sample of 1 values is taken from the each dataset

#Subsetting the data into clusters for 0-9, labelled data
df_list_30 = [x[1] for x in DIGITS_PCA_30.groupby(('label'),sort=True)]

for i in range(0,len(df_list_30)):
    df_list_30[i]= df_list_30[i].reset_index()
    df_list_30[i] =df_list_30[i].drop(['index'], axis=1)

center_list=[]
km_iterlist=[]

for i in range(0,10):
    center=df_list_30[i].sample(n=1, replace=True)
    center_list.append(center)
    center_list[i]=center_list[i].drop(['label'], axis=1)
    
clist=pd.concat(center_list)
#print(cl)
### The only difference in the code is the init parameter in the Kmeans algorithm , wherein the sample of 1 values is taken from the each dataset

km = KMeans(n_clusters=10,init=clist, n_init=1).fit(principalDf30)

y_kmeans = km.predict(principalDf30)
df_temp = pd.DataFrame(y,y_kmeans, columns=["True"]).reset_index()
df_temp.head(5)
majority_count_class= 0
for i in df_temp["index"].unique():
    df_inner = df_temp[df_temp["index"] == i]
    count_true_values = df_inner["True"].value_counts()
    print(count_true_values)
    majority_count_class += count_true_values.iloc[0]
        
majority_label = majority_count_class/42000
print(majority_label)   
   
iteration = km.n_iter_  
print(iteration)  


Purity_.loc[10]=np.array(["All",iteration,majority_label])

##Printing the table with the purity and iteration values for different initialization scenarios
Purity_
###6b

center_list=[]
kmiterlist=[]
kmeans_list=[5,10,15,20,25]

purity_list=[]
for i in range(0,len(kmeans_list)):
  
    #print(center_list[i])
    km = KMeans(n_clusters=kmeans_list[i],init="random", n_init=1).fit(principalDf30)
    
    #print(km.n_iter_)
    #kmiterlist.append(km.n_iter_)
    
    
    y_kmeans = km.predict(principalDf30)
    df_temp = pd.DataFrame(y,y_kmeans, columns=["True"]).reset_index()
    df_temp.head(5)

    majority_count_value= 0
    
    for i in df_temp["index"].unique():
        df_inner = df_temp[df_temp["index"] == i]
        count_true_val = df_inner["True"].value_counts()
        #print(count_true_val)
        majority_count_value += count_true_val.iloc[0]
        
    majority_purity_label = majority_count_value/42000
    #print(majority_purity_label)   
    purity_list.append(majority_purity_label)    
   
sns.scatterplot(kmeans_list,purity_list)
plt.title("K-values vs Purity")
import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/dataset.csv", sep=',')             #Read dataset. Ensure that dataset.csv is in the 

data = data_orig                                            #same location as the python notebook
data.replace('?', np.NaN, inplace= True)                    #Replace empty cells with NaN
#Fill cells having null value with the mean of the entire column for all numeric type columns 

data.fillna(data.mean(), inplace = True)                    
#For all categorical variables, fill empty cells with mode of the corresponding column

for column in ['Account1', 'History','Motive' ,'Monthly Period','History','Credit1','InstallmentRate','Sponsors','Tenancy Period','Age','InstallmentCredit','Yearly Period','Account2','Employment Period','Gender&Type','Sponsors','Plotsize','Plan','Housing','Post','Phone','Expatriate']:    

    data[column].fillna(data[column].mode()[0],inplace=True)
data = data.drop(labels = ['id'], axis = 1)                 #Drop id column as it is unique for all rows 

                                                             #and hence not useful
data.head()
#Find number of unique values for all categorical columns

for column in ['Account1','History','Motive','Account2','Employment Period','Gender&Type','Sponsors','Plan','Housing','Post']:

    print(data[column].unique())
#Since motive has a lot of unique values and it's one hot encoding will increase the dimension a lot so we drop it.

data = data.drop(labels = ['Motive'], axis = 1) 
#One-hot encoding of all categorical variables

data1 = data.copy()

data1 = pd.get_dummies(data1, columns=['Account1','Account2','Gender&Type','History','Sponsors','Plan','Housing','Employment Period', 'Post'])

data1.head()
data1.info()
#Converting all object type variables to float

data1['Monthly Period'] = data1['Monthly Period'].astype(float)

data1['Credit1'] = data1['Credit1'].astype(float)

data1['InstallmentRate'] = data1['InstallmentRate'].astype(float)

data1['#Credits'] = data1['#Credits'].astype(float)

data1['Tenancy Period'] = data1['Tenancy Period'].astype(float)

data1['Age'] = data1['Age'].astype(float)

data1['#Authorities'] = data1['#Authorities'].astype(float)

data1['Expatriate'] = data1['Expatriate'].astype(float)

data1['InstallmentCredit'] = data1['InstallmentCredit'].astype(float)

data1['Yearly Period'] = data1['Yearly Period'].astype(float)



data1.info()
data1['Phone'].unique()
data1['Phone'].size
#Changing categorical values to integer

for i in range(1031):

    if data1['Phone'][i] == 'yes':

        data1['Phone'][i] = 1

    if data1['Phone'][i] == 'no':

        data1['Phone'][i] = 0
data1['Plotsize'].unique()
#Changing categorical values to integer

for i in range(1031):

    if data1['Plotsize'][i] == 'XL':

        data1['Plotsize'][i] = 3

    if data1['Plotsize'][i] == 'LA' or data1['Plotsize'][i] == 'la':

        data1['Plotsize'][i] = 2

    if data1['Plotsize'][i] == 'ME' or data1['Plotsize'][i] == 'me' or data1['Plotsize'][i] == 'M.E.' :

        data1['Plotsize'][i] = 1

    if data1['Plotsize'][i] == 'sm' or data1['Plotsize'][i] == 'SM':

        data1['Plotsize'][i] = 0
#HeatMap for correlation assessment

import seaborn as sns

f, ax = plt.subplots(figsize=(30, 20))

corr = data1.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
#Dropping columns with correlation more than 0.25

data1 = data1.drop(labels = ['History_c4','Gender&Type_F0','Yearly Period','InstallmentCredit','Monthly Period','Post_Jb1','Post_Jb4','Employment Period_time5','Housing_H3'], axis = 1)
data1.head()
#Normalization 

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



scaler=MinMaxScaler()

scaled_data=scaler.fit(data1).transform(data1)

scaled_df=pd.DataFrame(scaled_data,columns=data1.columns)

scaled_df.tail()
data1.shape
scaled_df.shape
#PCA to reduce dimension

from sklearn.decomposition import PCA

pca = PCA().fit(scaled_df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance');
pca1 = PCA(n_components=30)

pca1.fit(scaled_df)

T1 = pca1.transform(scaled_df)
from sklearn.cluster import AgglomerativeClustering as AC

aggclus = AC(n_clusters = 3,affinity='euclidean',linkage='ward',compute_full_tree='auto')

y_aggclus= aggclus.fit_predict(scaled_df)

plt.scatter(T1[:, 0], T1[:, 1], c=y_aggclus)
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

from scipy.cluster.hierarchy import fcluster

linkage_matrix1 = linkage(scaled_df, "ward",metric="euclidean")

ddata1 = dendrogram(linkage_matrix1,color_threshold=10)
y_ac=cut_tree(linkage_matrix1, n_clusters = 3).T

y_ac
plt.scatter(T1[:,0], T1[:,1], c=y_ac[0,:], s=100, label=y_ac[0,:])

plt.show()
y_ac
y_ac = y_ac.tolist()

print(type(y_ac))
print(y_ac[0])
result = pd.DataFrame(y_ac[0])

final = pd.concat([data_orig["id"], result], axis=1).reindex()

final = final.rename(columns={0: "Class"})

final.head()
final.to_csv('2015B4A70708G_sub.csv', index = False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df,title = "Download csv file", filename = "data.csv"):

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href = "data:text/csv;base64,{payload}" target = "_blank">{title}</a>'

    html = html.format(payload=payload,title = title, filename = filename)

    return HTML(html)



create_download_link(final)
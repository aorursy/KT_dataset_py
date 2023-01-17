import sys

# !{sys.executable} -m pip install more-itertools

# !{sys.executable} -m pip install numpy

# !{sys.executable} -m pip install pandas

# !{sys.executable} -m pip install matplotlib

# !{sys.executable} -m pip install sklearn

import numpy as np

import pandas as pd

import matplotlib.pyplot as mp
data_orig = pd.read_csv("../input/dmassign1/data.csv", sep=',',na_values='?')

data = data_orig

data=data.set_index('ID')



cl=data["Class"]

data=data.drop(["Class"],1)



data.info()

data.head(10)
null_columns = data.isnull().sum().sort_values(ascending=False)

null_columns = null_columns[null_columns > 0]

null_columns

for column in null_columns.index:

    if(isinstance(data[column][0],str)):

        data[column].fillna(data[column].value_counts().index[0],inplace=True)

    elif(isinstance(data[column][0],float)):

        data[column].fillna(data[column].median(),inplace=True)# Have to discuss between mean or median

data.info()

gnull_columns = data.isnull().sum().sort_values(ascending=False)

gnull_columns = gnull_columns[gnull_columns > 0]

gnull_columns
print(data["Col189"].value_counts(normalize=True) * 100)

print(data["Col190"].value_counts(normalize=True) * 100)

print(data["Col191"].value_counts(normalize=True) * 100)

print(data["Col192"].value_counts(normalize=True) * 100)

print(data["Col193"].value_counts(normalize=True) * 100)

print(data["Col194"].value_counts(normalize=True) * 100)

print(data["Col195"].value_counts(normalize=True) * 100)

print(data["Col196"].value_counts(normalize=True) * 100)

print(data["Col197"].value_counts(normalize=True) * 100)
data["Col196"]=data["Col196"].str.lower()

data = data.replace({'Col196': {"m.e.":"me"}})
corr_matrix = data_orig.corr().abs()

pd.options.display.max_rows = None

corr_matrix["Class"].sort_values()



# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# to_drop = []

# for i in corr_matrix["Class"].index:

#     if(corr_matrix["Class"][i]<0.9):

#         to_drop.append(i)

# to_drop

# data_orig.drop(data[to_drop], axis=1)
# data = pd.get_dummies(data,columns=["Col189","Col190","Col191","Col192","Col193","Col194","Col195","Col196","Col197"])

# data.info()

data = data.drop(["Col189","Col190"],1)

data = data.drop(["Col191","Col192","Col193","Col194","Col195","Col196","Col197"],1)

data.head()
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data=scaler.fit_transform(data)

data=pd.DataFrame(scaled_data,columns=data.columns,index = data.index)

data.head()
from sklearn.decomposition import PCA



# pca1=PCA(n_components=2)

# pca1.fit(data)

# T1=pca1.fit_transform(data)

# mp.figure()

# mp.plot(np.cumsum(pca1.explained_variance_ratio_))

# mp.xlabel('Number of Components')

# mp.ylabel('Variance (%)') #for each component

# mp.title('Explained Variance Ratio')

# mp.show()

from sklearn.manifold import TSNE



model=TSNE(n_iter=1000,n_components=2,perplexity=100)

T1=model.fit_transform(data)
T1
from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix



wcss=[]

predictions=[]

correct=[]

for i in range(5, 60):

    kmean=KMeans(n_clusters = i,random_state=250)

    kmean.fit(T1)

    

    pr=kmean.predict(T1)

    forecast=pd.Series(pr+1,index=data.index,dtype = np.float64)

    classes=(confusion_matrix(cl[:1300],forecast[:1300]).argmax(axis=0)+1).astype(np.int64)

    forecast.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)

    non_matches=((forecast[:1300] != cl[:1300])).sum()

    matches=1300-non_matches

    

    correct.append(matches)

    predictions.append(forecast)

    wcss.append(kmean.inertia_)

    

    

mp.plot(range(5,60),wcss)

mp.title('The Elbow Method')

mp.xlabel('Number of clusters')

mp.ylabel('WCSS')

mp.show()



pred_index=correct.index(max(correct))

print("Matches: ",max(correct))

print("Accuracy: ",max(correct)*100/1300,"%")

print("Number of Clusters: ",pred_index+5)
#Converting float values to int

predictions[pred_index] = [int(i) for i in predictions[pred_index]]
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

output = pd.DataFrame(list(zip(data_orig["ID"][1300:13000], predictions[pred_index][1300:13000])), columns=['ID','Class'])

def create_download_link(data_orig, title = "Download CSV file", filename = "data.csv"): 

    csv = data_orig.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html =  '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(output)
import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

%matplotlib inline
# Reading Data

data = pd.read_csv("../input/dmassign1/data.csv",na_values='?',usecols = range(199))

df = pd.read_csv("../input/dmassign1/data.csv", sep=',')

check = data['Class']

data.drop(['Class','ID'],inplace=True,axis=1)

data.info()

data.head()
# Removing Null Values

col = data.isnull().sum().sort_values(ascending=True)

col = col[col > 0]

col
for column in col.index:

    if(isinstance(data[column][0],str)):

        data[column].fillna(data[column].value_counts().index[0],inplace=True)

    elif(isinstance(data[column][0],float)):

        data[column].fillna(data[column].median(),inplace=True)
data.info()
# Removing duplicate values

data.drop_duplicates()

data.info()
#Find percentage of distinct entries of certain Columns to ensure quality of data

print(data["Col189"].value_counts(normalize=True) * 100)

print(data["Col190"].value_counts(normalize=True) * 100)

print(data["Col191"].value_counts(normalize=True) * 100)

print(data["Col192"].value_counts(normalize=True) * 100)

print(data["Col193"].value_counts(normalize=True) * 100)

print(data["Col194"].value_counts(normalize=True) * 100)

print(data["Col195"].value_counts(normalize=True) * 100)

print(data["Col196"].value_counts(normalize=True) * 100)

print(data["Col197"].value_counts(normalize=True) * 100)
data["Col197"]=data["Col197"].str.lower()

data = data.replace({'Col197': {"m.e.":"me"}})

print(data["Col197"].value_counts(normalize=True) * 100)
data.info()
#dimension reduction

data=data.drop(["Col189","Col193","Col196","Col195","Col194","Col191","Col190","Col197","Col192"],1)
data.info()
# Normalization

scaler=StandardScaler()

np_scaled=scaler.fit_transform(data)

np_scaled=pd.DataFrame(np_scaled,columns=data.columns)

data = pd.DataFrame(scaler.transform(data),columns=data.columns)
# Dimension Reduction

from sklearn.manifold import TSNE

model=TSNE(n_iter=10000,n_components=2,perplexity=100)

T1=model.fit_transform(data)

T2=T1
# for loop to determine the number of clusters

from sklearn.cluster import KMeans



wcss = []

some = []

highest=0

number=0

for i in range(5, 50):

    kmean = KMeans(n_clusters = i, random_state = 30)

    kmean.fit(T2)

    pred = kmean.predict(T1)

    predictions = pd.Series(pred+1,index=data.index,dtype = np.float64)

    classes = (confusion_matrix(check[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)

    predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)

    tot = ((predictions[:1300] != check[:1300])).sum()

    some.append(tot)

    wcss.append(kmean.inertia_)

    count=0

    for j in range(1300):

        if int(predictions[j]) == int(check[j]):

            count = count + 1

    if(count>highest):

        highest=count

        number=i

    print(i)

    

print ("Number of matches in Test Data: ",highest)

print ("Percentage Accuracy: ",highest*100/1300)

print ("Number of Cluster for best result",number)
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

plt.xlabel('Number of clusters')

plt.ylabel('Unequal values')

plt.plot(range(5,50),some)
plt.plot(range(5,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
# Final application of K-Means with 48 clusters and post-processing using confusion matrix

kmean = KMeans(n_clusters = 49, random_state = 30)

kmean.fit(T2)

pred = kmean.predict(T1)

predictions = pd.Series(pred+1,index=data.index,dtype = np.float64)

classes = (confusion_matrix(check[:1300],predictions[:1300]).argmax(axis=0)+1).astype(np.float64)

predictions.replace({cluster+1:classes[cluster] for cluster in range(0,len(classes))},inplace=True)
# Accuracy calculation on the basis of first 1300 entries

count=0

for j in range(1300):

    if int(predictions[j]) == int(check[j]):

        count = count + 1

print ("Number of matches in Test Data: ",count)

print ("Percentage Accuracy: ",count*100/1300)
predictions[:5]
list1=predictions.astype(int).tolist()
list1[:5]
plt.figure(figsize=(16, 8))

plt.title("49 clusters")

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
# CSV file

from IPython.display import HTML 

import pandas as pd

import numpy as np

import base64

output=pd.DataFrame(list(zip(df["ID"][1300:13000],list1[1300:13000])),columns=['ID','Class'])

def create_download_link(df, title = "Download CSV file", filename = "2016A7PS0077G_check.csv"): 

    csv = output.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(output)
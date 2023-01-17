import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Import and Process the data

data_orig = pd.read_csv("../input/dataset.csv", sep=',')

data = data_orig

#Training Data is stored in 'check'

check = (data["Class"])[0:175]

ids = data["id"]

#Find and Replace the missing values from the data set with the number -1

data = data.replace('?',-1)

data.info()

data.head(10)
data["Monthly Period"] = data["Monthly Period"].astype(int)

data["Credit1"] = data["Credit1"].astype(int)

data["InstallmentRate"] = data["InstallmentRate"].astype(int)

data["Tenancy Period"] = data["Tenancy Period"].astype(int)

data["Age"] = data["Age"].astype(int)

data["InstallmentCredit"] = data["InstallmentCredit"].astype(float)

data["Yearly Period"] = data["Yearly Period"].astype(float)

data.info()
#Find percentage of distinct entries of certain Columns to ensure quality of data

print(data["Plotsize"].value_counts(normalize=True) * 100)

print(data["Sponsors"].value_counts(normalize=True) * 100)

print(data["Account1"].value_counts(normalize=True) * 100)

print(data["Tenancy Period"].value_counts(normalize=True) * 100)

print(data["Motive"].value_counts(normalize=True) * 100)

print(data["History"].value_counts(normalize=True) * 100)
#Preprocess the data based on observations from the data

data = data.replace({'Account1': {-1:"ad"}})

data = data.replace({'Monthly Period': {-1:21}})

data = data.replace({'History': {-1:"c4"}})

data = data.replace({'Motive': {-1:"p0"}})

data = data.replace({'Credit1': {-1:3251}})

data = data.replace({'InstallmentRate': {-1:4}})

data = data.replace({'Account2': {-1:"sacc1"}})

data = data.replace({'Employment Period': {-1:"time3"}})

data = data.replace({'Tenancy Period': {-1:4}})

data = data.replace({'#Credits': {4:1}})

data = data.replace({'Age': {-1:25}})

data = data.replace({'Post': {"Jb1":"Jb3"}})

data["Plotsize"]=data["Plotsize"].str.lower()

data["Sponsors"]=data["Sponsors"].str.lower()

data = data.replace({'Plotsize': {"m.e.":"me"}})



#The value '0' is replaced by the mean of the existing data to reduce outliers

mean1 = data["InstallmentCredit"].mean() #Calculate Mean of Installment Credit

mean2 = data["Yearly Period"].mean() #Calculate Mean of Yearly Period

data = data.replace({'InstallmentCredit': {-1:mean1}})

data = data.replace({'Yearly Period': {-1:mean2}})

data.head(10)
#On the basis of the domain to which the data belongs, attributes are selected to find the best possible results

#"Class" is dropped since it is to be predicted

data = data.drop(["id", "Employment Period", "Post", "Plan"],1)

data = data.drop(["History", 'Phone', "Gender&Type", 'Account2', "Sponsors", "Class"],1)

data = pd.get_dummies(data, columns=['Account1', "Housing", "Plotsize", "Tenancy Period", "Expatriate", "Motive"])

data.info()
#Min-max scaling using sklearn

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(data)

data = pd.DataFrame(np_scaled)

data.head(5)
#Principal Component Analysis (PCA) using sklearn

from sklearn.decomposition import PCA

pca1 = PCA(0.95)

pca1.fit(data)

T1 = pca1.transform(data)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

wcss = []

for i in range(2, 10):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(data)

    wcss.append(kmean.inertia_)

plt.figure(figsize=(5, 5))

plt.plot(range(2,10),wcss)

plt.title('The Elbow Method (Cluster Analysis)')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

# plt.show() #Uncomment to show graph



#Dynamic Clusters and Prediction Data

#Only diagram for 3 clusters has been shown as it produced the most accurate prediction

plt.figure(figsize=(35, 10))

preds2 = []

for i in range(3,4):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(data)

    pred = kmean.predict(data)

    preds2.append(pred)

#     plt.subplot(2, 4, i - 1)

#     plt.title(str(i)+" clusters")

#     plt.scatter(T1[:, 0], T1[:, 1], c=pred)

#     centroids = kmean.cluster_centers_

#     centroids = pca1.transform(centroids)

#     plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown')
#Iterate through all possible combinations of clusters and classes to find the most accurate value

listnum = [0,1,2]

from itertools import permutations

list2 = []

for combo in permutations(listnum, 3):

    list2.append(combo)

    

#list1 contains predicted values for various combinations

#count1 contains the number of matches for each combination

list1 = []

count1 = []

for i in range(len(list2)):

    a=list2[i][0]

    b=list2[i][1]

    c=list2[i][2]

    count = 0

    temp = []

    for i in range(len(pred)):

        if pred[i] == a:

            temp.append(0)

        elif pred[i] == b:

            temp.append(1)

        elif pred[i] == c:

            temp.append(2)

    list1.append(temp)

    for i in range(0,175):

        if int(temp[i]) == int(check[i]):

            count = count + 1

    count1.append(count)

print ("Number of matches in Test Data: ",max(count1))

print ("Percentage Accuracy: ",max(count1)*100/175)

#These are the final 856 "Class" values for the test data



# print ("Numbers to be stored in the submission file: ")

# for i in range(175,1031):

#     print (list1[count1.index(max(count1))][i])
out = list1[count1.index(max(count1))][175:1031]

final = pd.DataFrame({"id":ids[175:1031],"Class":out})

final.to_csv("finalsub.csv", index=False)

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(final)
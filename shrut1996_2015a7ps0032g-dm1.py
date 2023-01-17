# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from scipy.spatial.distance import cdist

from sklearn.cluster import KMeans

from collections import Counter
df = pd.read_csv('../input/dataset.csv')

df = df.replace('?',np.NaN)

df.head()
df['Account1'].fillna('aa', inplace = True) 

df['Monthly Period'].fillna('36', inplace = True) 

df['History'].fillna('c4', inplace = True) 

df['Motive'].fillna('p0', inplace = True) 

df['Credit1'].fillna('1500', inplace = True) 

df['InstallmentRate'].fillna('4', inplace = True) 

df['Tenancy Period'].fillna('4', inplace = True) 

df['Age'].fillna('30', inplace = True) 

df['InstallmentCredit'].fillna('0.0', inplace = True) 

df['Yearly Period'].fillna('0.0', inplace = True) 

label = df['Class'].iloc[:175]

data = df.drop(['id', 'Class'], 1)

data.dropna(inplace=True)



object_to_int_columns = ['Monthly Period', 'Credit1', 'Age']



for i in object_to_int_columns:

    data[i]=data[i].astype(str).astype(int)

object_to_float_columns = ['InstallmentCredit', 'Yearly Period']



for i in object_to_float_columns:

    data[i]=data[i].astype(str).astype(float)

    

data.info()

# for i in range(1, len(data.columns)):

#     print(data.iloc[:,i].unique())
data['Account2'].replace({'Sacc4':'sacc4'},inplace=True)

data['Sponsors'].replace({'g1':'G1'},inplace=True)

data['Plotsize'].replace({'la':'LA', 'sm':'SM','me':'ME', 'M.E.':'ME'},inplace=True)

data['Phone'].replace({'no':0, 'yes':1},inplace=True)

data['Expatriate'].replace({False:0, True:1},inplace=True)

categorical_variables = ['Account1', 'History', 'Motive', 'Account2', 'Employment Period', 'InstallmentRate', 'Gender&Type', 'Sponsors',

                         'Tenancy Period', 'Plotsize', 'Plan', 'Housing', 'Post']

data1 = pd.get_dummies(data, columns=categorical_variables)



# data1.head()
# corr = data1.corr(method="kendall")



# # Generate a mask for the upper triangle

# mask = np.zeros_like(corr, dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True



# # Set up the matplotlib figure

# f, ax = plt.subplots(figsize=(22, 18))



# # Generate a custom diverging colormap

# cmap = sns.diverging_palette(220, 10, as_cmap=True)



# # Draw the heatmap with the mask and correct aspect ratio

# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

#             square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)

data1 = data1.drop(['InstallmentCredit', 'Yearly Period'], 1)

scaled_data = StandardScaler().fit_transform(data1)

scaled_df=pd.DataFrame(scaled_data,columns=data1.columns)

scaled_df.head()
pca = PCA(n_components=40)

pca.fit(scaled_df)

T1 = pca.transform(scaled_df)

pca.explained_variance_ratio_.sum()
# distortions = []

# K = range(1,13)

# for k in K:

#     kmeanModel = KMeans(n_clusters=k, n_init=50, random_state=42).fit(X)

#     kmeanModel.fit(X)

#     distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])



# # Plot the elbow

# plt.plot(K, distortions, 'bx-')

# plt.xlabel('k')

# plt.ylabel('Distortion')

# plt.title('The Elbow Method showing the optimal k')

# plt.show()
mx, mr, mn = 0, 0, 0



# Random centroid initialization and n_clusters simulation

for r in range(1,50):

    for num in range(6, 13):

        kmeans = KMeans(n_clusters = num, random_state=r)

        pred = kmeans.fit_predict(T1)



        a = {}

        for item in range(num):

            a[item] = []

        

        for index, p in enumerate(pred[:175]):

            a[p].append(index)



        subs = {}

        for item in range(num):

            subs[item] = int(Counter(df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



        test = [subs.get(n, n) for n in pred[:175]]

        pred1 = [subs.get(n, n) for n in pred[175:]]



        correct, total = 0,0

        for i,j in zip(test, label[:175]):

            if i==int(j):

                correct+=1

            total+=1



        if correct/total>mx:

            mx = correct/total

            mn = num

            mr = r

    print('Iteration :', r)

    

print('Found optimal hyperparameters ->')

print('Number of clusters: ', mn)

print('Random State: ', mr)

kmeans = KMeans(n_clusters = mn, random_state=mr)

pred = kmeans.fit_predict(T1)



plt.figure(figsize=(16, 8))

plt.scatter(T1[:, 0], T1[:, 1], c=pred)
a = {}

for item in range(mn):

    a[item] = []



for index, p in enumerate(pred[:175]):

    a[p].append(index)

    

subs = {}

for item in range(mn):

    subs[item] = int(Counter(df['Class'].iloc[a[item]].dropna()).most_common(1)[0][0])



test = [subs.get(n, n) for n in pred[:175]]

pred1 = [subs.get(n, n) for n in pred[175:]]
correct, total = 0,0

for i,j in zip(test, label[:175]):

    if i==int(j):

        correct+=1

    total+=1



print(correct/total)
penul = pd.DataFrame({'id': df['id'].iloc[175:], 'Class': pred1})



len(penul)

penul.head()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = penul.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)





# create a link to download the dataframe

create_download_link(penul)
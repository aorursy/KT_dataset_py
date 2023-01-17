import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
data_orig = pd.read_csv("../input/dataset.csv", na_values = '?')

data = data_orig
import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))

corr = data.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
cols= data.columns.values
cols
data.head()
preds = data['Class']
ids = data['id']
ids[:5]
data = data.drop('id',axis = 1)
data = data.drop(['Credit1','Monthly Period'], axis = 1)
data.info()
cols = data.columns.values
vals = []

for i in cols:

    vals = data[i].unique()

    print("Column ", i , " has values ", vals)
data['Account2'] = data['Account2'].replace('Sacc4','sacc4')
data['Sponsors'] = data['Sponsors'].replace('g1','G1')

data['Plotsize'] = data['Plotsize'].replace('sm','SM')

data['Plotsize'] = data['Plotsize'].replace('me','ME')

data['Plotsize'] = data['Plotsize'].replace('M.E.','ME')

data['Plotsize'] = data['Plotsize'].replace('la','LA')
vals = []

for i in cols:

    vals = data[i].unique()

    print("Column ", i , " has values ", vals)
data = data.drop(['Class'], axis = 1)
data = data.drop(['Motive','Account2', 'Employment Period', 'InstallmentRate', 'Gender&Type','Sponsors', 'Tenancy Period', 'Plan','#Credits','Post','#Authorities','Phone','Expatriate', 'InstallmentCredit'], axis = 1 )
cat_cols = ['Account1','History','Plotsize','Housing','Age']
num_cols = ['Yearly Period']
#['Motive','Account2', 'Employment Period', 'InstallmentRate', 'Gender&Type', 'Monthly Period','Sponsors', 'Tenancy Period', 'Plan','#Credits','Post','#Authorities','Phone','Expatriate', 'InstallmentCredit'] 
for i in cat_cols:

    data[i]= data[i].fillna(data[i].mode()[0])
for i in num_cols:

    data[i]= data[i].fillna(data[i].mean())
data.info()
cat_cols = ['Account1','History','Plotsize','Housing','Age']
one_hot = pd.get_dummies(data[cat_cols])
one_hot.info()
data.drop(cat_cols,axis = 1, inplace = True)
data= data.join(one_hot)
data.info()
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
data.head()
scalar = MinMaxScaler()
cols = data.columns.values
data = pd.DataFrame(scalar.fit_transform(data),columns = data.columns)
train = data[0:175]
test = data[175:]
test.shape
target = preds
from sklearn.cluster import KMeans



wcss = []

for i in range(2, 50):

    kmean = KMeans(n_clusters = i, random_state = 42)

    kmean.fit(data)

    wcss.append(kmean.inertia_)

    

plt.plot(range(2,50),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
train.info()
from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)

pca1.fit(data)

T1 = pca1.transform(data)
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score

kmean = KMeans(n_clusters = 3, random_state = 91)

kpred = kmean.fit_predict(data)
target
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
def combination_best_accuracy(pred,tar):

    acc = 0 

    checkval = pd.DataFrame(pred[:175],columns=['values'])

    tar = tar.loc[:174]

    combi = [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]

    pred = pd.DataFrame(pred,columns=['values'])

    outval = pred

    

    for i, comb in enumerate(combi):

        pr_temp = checkval['values'].apply(lambda x: comb[0] if x==0 else comb[1] if x==1 else comb[2])

        acc_temp = accuracy_score(pr_temp,tar)

        if(acc_temp>acc):

            acc = acc_temp

            outval = pred['values'].apply(lambda x: comb[0] if x==0 else comb[1] if x==1 else comb[2])

            

    return acc*100,outval.as_matrix()
bestacc, outvals = combination_best_accuracy(kpred, target)
bestacc
outvals.shape
tpred = outvals.tolist()

tpred = tpred[:175]

tpred.count(0), tpred.count(1), tpred.count(2)
ttarg = target.tolist()

ttarg= ttarg[:175]

ttarg.count(0), ttarg.count(1), ttarg.count(2)
predplot = kpred.astype(float)

for i in range(len(predplot)):

    if i > 175:

        predplot[i]= np.nan


plt.figure(figsize=[10,10])



plt.subplot(2, 1, 1)

plt.title("Predicted Cluster using KMeans")

plt.scatter(T1[:, 0], T1[:, 1], c=predplot)



centroids = kmean.cluster_centers_

centroids = pca1.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)



plt.subplot(2, 1, 2)

plt.title("Visualisation of the Actual Cluster")

plt.scatter(T1[:, 0], T1[:, 1], c=target)



centroids = kmean.cluster_centers_

centroids = pca1.transform(centroids)

plt.plot(centroids[:, 0], centroids[:, 1], 'b+', markersize=30, color = 'brown', markeredgewidth = 3)





zero, one, two = 0,0,0

for i in range(len(ttarg)):

    if(ttarg[i]==0 and tpred[i]==0):

        zero+=1

    elif (ttarg[i]==0 and tpred[i]==1):

        one+=1

    elif (ttarg[i]==0 and tpred[i]==2):

        two+=1

print(zero, one, two)
zero, one, two = 0,0,0

for i in range(len(ttarg)):

    if(ttarg[i]==1 and tpred[i]==0):

        zero+=1

    elif (ttarg[i]==1 and tpred[i]==1):

        one+=1

    elif (ttarg[i]==1 and tpred[i]==2):

        two+=1

print(zero, one, two)
zero, one, two = 0,0,0

for i in range(len(ttarg)):

    if(ttarg[i]==2 and tpred[i]==0):

        zero+=1

    elif (ttarg[i]==2 and tpred[i]==1):

        one+=1

    elif (ttarg[i]==2 and tpred[i]==2):

        two+=1

print(zero, one, two)
outvals = outvals[175:]
len(outvals)
data_tst = pd.read_csv('../input/sample_submission.csv')

data_tst['Class'] = outvals

data_tst[['id','Class']].to_csv(f'sub12.csv',index = False)
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



create_download_link(data_tst[['id','Class']])
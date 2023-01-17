import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm as tq

from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler, RobustScaler

%matplotlib inline
df = pd.read_csv("train.csv")
tdf = pd.read_csv("test.csv")
df.loc[df.tenure==0,'TotalCharges']='0'

tdf.loc[tdf.tenure==0,'TotalCharges']='0'
def tenure_conv(df) :

    return "Tenure"+str(df["tenure"]/12)
df["tenure"] = df.apply(lambda x:tenure_conv(x),

                                      axis = 1)

tdf["tenure"] = tdf.apply(lambda x:tenure_conv(x),

                                      axis = 1)


df['Married'] = df['Married'].eq('Yes').mul(1)

df['Children'] = df['Children'].eq('Yes').mul(1)

df['Internet'] = df['Internet'].eq('Yes').mul(1)

df['AddedServices'] = df['AddedServices'].eq('Yes').mul(1)





tdf['Married'] = tdf['Married'].eq('Yes').mul(1)

tdf['Children'] = tdf['Children'].eq('Yes').mul(1)

tdf['Internet'] = tdf['Internet'].eq('Yes').mul(1)

tdf['AddedServices'] = tdf['AddedServices'].eq('Yes').mul(1)

df['gender'] = df['gender'].eq('Male').mul(1)

tdf['gender'] = tdf['gender'].eq('Male').mul(1)

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for i in ohcols:

    df[i] = le.fit_transform(df[i])

    tdf[i] = le.fit_transform(tdf[i])

    
df.head()
already = ['Married','Children','Internet','AddedServices','gender']

ohcols = ['tenure','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','HighSpeed','Subscription','PaymentMethod']
onehot = pd.get_dummies(data=df, columns = ohcols,prefix = ohcols)

onehot

df = df.merge(onehot)

onehot = pd.get_dummies(data=tdf, columns = ohcols,prefix = ohcols)

onehot

tdf = tdf.merge(onehot)
df = df.drop(ohcols,axis=1)

tdf = tdf.drop(ohcols,axis=1)

scaler = StandardScaler()

df[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['MonthlyCharges', 'TotalCharges']])

tdf[['MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(tdf[['MonthlyCharges', 'TotalCharges']])

df.info()
numerical_features = df.columns.difference(['custId','Satisfied'])

numerical_features
X = df[numerical_features]

y = df["Satisfied"]

X_test = tdf[numerical_features]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components=5)

X = lda.fit_transform(X, y)

X_test = lda.transform(X_test)
os = df[df.Satisfied==0]

df = df.append(os)

df = df.append(os)
df.Satisfied.value_counts()
from sklearn.cluster import KMeans

from sklearn.metrics import roc_auc_score

mi = 0



for nc in tq(range(3,6)):

    n_clusters = nc

    for mask in range(2**n_clusters):

            clf = KMeans(n_clusters = n_clusters, random_state=42).fit(X)

            y_pred = clf.predict(X)

            cluster_out = [0]*n_clusters

            for i in range(n_clusters):

                if mask&(1<<i)!=0:

                    cluster_out[i] = 1

            

            y_pred = clf.predict(X_test)

            for i in range(len(y_pred)):

                y_pred[i] = cluster_out[y_pred[i]]

            acc = roc_auc_score(y,y_pred)

            if acc > mi:

                print(acc,mask)

                mi = acc

                bi = mask
print(acc)
bi
from sklearn.decomposition import PCA

pca = PCA(n_components=40)

principalComponents = pca.fit_transform(X)

principalDf = pd.DataFrame(data = principalComponents)

testpdf = pd.DataFrame(data = pca.transform(X_test))
n_clusters = 167

clf = KMeans(n_clusters = n_clusters, random_state=42).fit(principalDf)

y_labels_train = clf.labels_

y_pred = clf.predict(principalDf)



cluster_out = {}

ones = [0]*n_clusters

zeros = [0]*n_clusters



for i in range(len(principalDf)):

    if y.iloc[i] == 0:

        zeros[y_pred[i]] +=1

    else:

        ones[y_pred[i]] +=1

for i in range(n_clusters):

    if ones[i]>zeros[i]:

        cluster_out[i] = 1

    else:

        cluster_out[i] = 0

y_pred = clf.predict(testpdf)

for i in range(len(y_pred)):

    y_pred[i] = cluster_out[y_pred[i]]
n_clusters = 6

mask = 49

clf = KMeans(n_clusters = n_clusters,random_state=42).fit(X)

y_pred = clf.predict(X)

cluster_out = [0]*n_clusters

for i in range(n_clusters):

    if mask&(1<<i)!=0:

        cluster_out[i] = 1

y_pred = clf.predict(X_test)

for i in range(len(y_pred)):

    y_pred[i] = cluster_out[y_pred[i]]

acc = roc_auc_score(y,y_pred)

print(acc)
n_clusters = 4

mask = 10

clf = KMeans(n_clusters = n_clusters,random_state=42).fit(X)

y_pred = clf.predict(X)

cluster_out = [0]*n_clusters

for i in range(n_clusters):

    if mask&(1<<i)!=0:

        cluster_out[i] = 1

y_pred = clf.predict(X_test)

for i in range(len(y_pred)):

    y_pred[i] = cluster_out[y_pred[i]]

acc = roc_auc_score(y,y_pred)

print(acc)
n_clusters = 6

mask = 49

mi = 0

for k in range(0,100):

    print(k,end=' ')

    clf = KMeans(n_clusters = n_clusters,random_state=k).fit(X)

    y_pred = clf.predict(X)

    cluster_out = [0]*n_clusters

    for i in range(n_clusters):

        if mask&(1<<i)!=0:

            cluster_out[i] = 1

    y_pred = clf.predict(X_test)

    for i in range(len(y_pred)):

        y_pred[i] = cluster_out[y_pred[i]]

    acc = roc_auc_score(y,y_pred)

    if acc>mi:

        mi = acc

        print(acc,k)
acc = roc_auc_score(y,y_pred)

print(acc)
upload = pd.concat([tdf.custId,pd.DataFrame(data=y_pred)],axis=1)

upload.columns = ['custId','Satisfied']

upload.to_csv('submit.csv',index=False)
from sklearn.ensemble import ExtraTreesClassifier

clf = ExtraTreesClassifier().fit(X,y)

y_pred = clf.predict(X_test)

acc = roc_auc_score(y,y_pred)

acc
lo = clf.predict([df[numerical_features].iloc[5]])

print(lo)
from sklearn.metrics import roc_auc_score



acc = roc_auc_score(y,y_pred)
acc
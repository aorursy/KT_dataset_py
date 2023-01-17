import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,auc,accuracy_score

from imblearn.over_sampling import SMOTE

from collections import Counter

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch



from sklearn.preprocessing import StandardScaler, normalize, RobustScaler, MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.metrics import silhouette_score,roc_curve
df = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv")

df.loc[df['TotalCharges'] == " ", "TotalCharges"] = df["MonthlyCharges"]

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

df.head()
def binarize(df):

    

    for col in ['gender']:

        df[col] = df[col].map({'Male':0, 'Female':1})



    for col in ['Married']:

        df[col] = df[col].map({'No':0, 'Yes':1})

        

    for col in ['Children']:

        df[col] = df[col].map({'No':0, 'Yes':1})

    

    for col in ['TVConnection']:

        df[col] = df[col].map({'No':0, 'Cable':1, 'DTH': 2})

        

    for col in ['Channel1']:

        df[col] = df[col].map({'No':0, 'Yes':1, 'No tv connection': 2})

        

    for col in ['Channel2']:

        df[col] = df[col].map({'No':0, 'Yes':1, 'No tv connection': 2})

        

    for col in ['Channel3']:

        df[col] = df[col].map({'No':0, 'Yes':1, 'No tv connection': 2})

        

    for col in ['Channel4']:

        df[col] = df[col].map({'No':0, 'Yes':1, 'No tv connection': 2})

        

    for col in ['Channel5']:

        df[col] = df[col].map({'No':0, 'Yes':1, 'No tv connection': 2})

        

    for col in ['Channel6']:

        df[col] = df[col].map({'No':0, 'Yes':1, 'No tv connection': 2})

    

    for col in ['Internet']:

        df[col] = df[col].map({'No':0, 'Yes':1})

        

    for col in ['HighSpeed']:

        df[col] = df[col].map({'No':0, 'Yes':1, 'No internet': 2 })

        

    for col in ['AddedServices']:

        df[col] = df[col].map({'No':0, 'Yes':1})

        

    for col in ['Subscription']:

        df[col] = df[col].map({'Monthly':0, 'Annually':1, 'Biannually': 2 })

        

    for col in ['PaymentMethod']:

        df[col] = df[col].map({'Net Banking':0, 'Cash':1, 'Bank transfer': 2, 'Credit card': 3 })

        

    

    return df
scale = RobustScaler()

X = df.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Children','Married','Satisfied'], axis=1)

y1 = df['Satisfied'].values

X = pd.get_dummies(X)

X = X

X = scale.fit_transform(X)

X = normalize(X)
# df = binarize(df)

# df_test = binarize(df_test)
# df.head()
# df_test.head()
# df = df.values

# df_test = df_test.values
# sm = SMOTE()

# #X = pd.get_dummies(df.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Satisfied','Children','Married','MonthlyCharges'], axis=1))

# X = pd.get_dummies(df.drop(['custId', 'gender','Satisfied','Internet','HighSpeed'], axis=1))

# # X = df.drop(['custId', 'gender','Satisfied'], axis=1).values

# y = df['Satisfied']

# X_res, y_res = sm.fit_resample(X, y)

# print('Resampled dataset shape %s' % Counter(y_res))
# scale = StandardScaler()

# X = scale.fit_transform(X)



# X = normalize(X) 
# df1 = df.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Satisfied','Children','Married','MonthlyCharges'], axis=1)

# df1 = pd.get_dummies(df1)
# corr = df.corr()

# corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# pca = PCA(n_components = 3) 

# X_principal = pca.fit_transform(df1) 

# X_principal = pd.DataFrame(X_principal) 

# X_principal.columns = ['P1', 'P2','P3'] 
# sse = {}

# for k in range(1, 10):

#     kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_principal)

#     sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

# plt.figure()

# plt.plot(list(sse.keys()), list(sse.values()))

# plt.xlabel("Number of cluster")

# plt.ylabel("SSE")

# plt.show()

# for i in range(1,10):

#     pca = PCA(n_components = i) 

#     X_principal = pca.fit_transform(X_res) 

#     X_principal = pd.DataFrame(X_principal) 

#     kmeans = KMeans(n_clusters=2, max_iter=5000).fit(X_principal)

#     labels= kmeans.labels_

#     count = 0

#     for i in range(len(labels)):

#         if labels[i]!=y[i]:

#             count = count+1

#     print(count)
# pca = PCA(n_components = 3) 

# X_principal = pca.fit_transform(X_res) 

# X_principal = pd.DataFrame(X_principal) 

# # X_principal.columns = ['P1', 'P2','P3'] 
# X_principal = X_principal.values
# kmeans = KMeans(n_clusters=2,random_state=42)

# kmeans.fit(X_principal)
# X = pd.get_dummies(df.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Satisfied','Children','Married','MonthlyCharges'], axis=1)).values

# # X = df.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','Satisfied'], axis=1).values

# y = df['Satisfied'].values

# sm = SMOTE()

# X_res, y_res = sm.fit_resample(X, y)

# # X_res, y_res = X, y

# print('Resampled dataset shape %s' % Counter(y_res))
df_test = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv")

df_test.loc[df_test['TotalCharges'] == " ", "TotalCharges"] = df_test["MonthlyCharges"]

df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'])



df_test.head()
scale = RobustScaler()

X_test = df_test.drop(['custId', 'gender','TVConnection','Internet','HighSpeed','SeniorCitizen','Children','Married'], axis=1)

X_test = pd.get_dummies(X_test)

X_test = X_test

X_test = scale.fit_transform(X_test)

X_test = normalize(X_test)
X_test.shape
for x in [0.4,0.5,0.6,0.7,0.8]:

    for y in [30,40,50,60,70,80]:

        algo = Birch(n_clusters=2,threshold = x, branching_factor=y).fit(X_test)

        labels = algo.labels_

        

        pred = algo.predict(X)

        fpr, tpr, thresholds = roc_curve(y1, pred, pos_label=0)

        print(x,y,"-->",auc(fpr, tpr), accuracy_score(y1, pred))
for x in [0.45,0.47,0.5,0.53,0.56]:

    for y in [25,28,30,32,31,27,35,38]:

        algo = Birch(n_clusters=2,threshold = x, branching_factor=y).fit(X_test)

        labels = algo.labels_

        

        pred = algo.predict(X)

        fpr, tpr, thresholds = roc_curve(y1, pred, pos_label=0)

        print(x,y,"-->",auc(fpr, tpr), accuracy_score(y1, pred))
# algo = KMeans(n_clusters=2).fit(X_test)  --> 0.599

# pred = algo.predict(X_test)

algo = Birch(n_clusters=2).fit(X_test)

#algo = Birch(n_clusters=2,threshold = 0.5, branching_factor=32).fit(X_test)

labels = algo.labels_

# count = 0

# #print(y_res.shape, labels.shape)

# for i in range(len(labels)):

#     if labels[i]!=y_res[i]:

#         count = count+1

        

# print(count)
labels
# algo = AgglomerativeClustering(n_clusters=2).fit(X_res)

# #pred = algo.predict(X_test)

# algo = AgglomerativeClustering.set_params(get_params(algo))

# labels = algo.labels_

# count = 0

# #print(y_res.shape, labels.shape)

# for i in range(len(labels)):

#     if labels[i]!=y_res[i]:

#         count = count+1

        

# print(count)
# algo = AgglomerativeClustering(n_clusters=2).fit(X_test)

# labels = algo.labels_
# labels.shape
# algo = SpectralClustering(n_clusters=2).fit(X_res)

# pred = algo.predict(X_test)



# labels = algo.labels_

# count = 0

# #print(y_res.shape, labels.shape)

# for i in range(len(labels)):

#     if labels[i]!=y_res[i]:

#         count = count+1

        

# print(count)
# algo2 = Birch(2).fit(X_res)

# pred2 = algo.predict(X_test)



# labels = algo2.labels_

# count = 0

# #print(y_res.shape, labels.shape)

# for i in range(len(labels)):

#     if labels[i]!=y_res[i]:

#         count = count+1

        

# print(count)
# X_test = pca.transform(X_test)

# pred = kmeans.predict(X_test)
# labels = 1-labels
result = pd.DataFrame({'Satisfied':labels}, index=df_test['custId'])

result.to_csv("sub9.csv")
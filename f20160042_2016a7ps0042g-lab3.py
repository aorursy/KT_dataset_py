import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

np.random.seed(42)
train = pd.read_csv("/kaggle/input/eval-lab-3-f464/train.csv", sep=',')
y_train = train['Satisfied']
y_train.size
train.head()
train['AddedServices'].unique()
train = train.drop(['Satisfied','custId'],axis=1)
train['TotalCharges']=train['TotalCharges'].replace(" ",'0')
train['TotalCharges']=train['TotalCharges'].astype(float)
cat = ['TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','HighSpeed','Subscription','PaymentMethod','gender','Married','Children','Internet','AddedServices']
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

for i in cat :

    train[i]=labelencoder.fit_transform(train[i])
train.head()
test = pd.read_csv("/kaggle/input/eval-lab-3-f464/test.csv", sep=',')

test = test.drop(['custId'],axis=1)

test['TotalCharges']=test['TotalCharges'].replace(" ",'0')

test['TotalCharges']=test['TotalCharges'].astype(float)

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

for i in cat :

    test[i]=labelencoder.fit_transform(test[i])
test.head()
train.isnull().sum()
from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()

np_scaled = std_scaler.fit_transform(train)

train = pd.DataFrame(np_scaled)

train.head()
np_scaled_test = std_scaler.transform(test)

test = pd.DataFrame(np_scaled_test)

test.head()
import seaborn as sns

f, ax = plt.subplots(figsize=(50, 40))

corr = train.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

            square=True, ax=ax, annot = True);
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA()

train = lda.fit_transform(train, y_train)

test = lda.transform(test)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2,random_state=1234)

kmeans.fit(train)

pred_train = kmeans.predict(train)



cluster_lab = {}

for i in range(len(pred_train)):

    if pred_train[i] not in cluster_lab:

        cluster_lab[pred_train[i]] = [y_train[i]]

    else:

        cluster_lab[pred_train[i]].append(y_train[i])

 

actual_lab={}        

for i in cluster_lab:

    actual_lab[i] = max(set(cluster_lab[i]), key=cluster_lab[i].count)    

 

for i in range(len(pred_train)):

    pred_train[i] = actual_lab[pred_train[i]]
from sklearn import metrics

metrics.roc_auc_score(y_train,pred_train)
y_test = kmeans.predict(test)

for i in range(len(y_test)):

    y_test[i] = actual_lab[y_test[i]]
test_final = pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

id_test = test_final['custId']

out = pd.concat([id_test,pd.DataFrame(y_test)],axis=1)

out.columns = ['custId','Satisfied']

out.to_csv('final.csv',index=False)
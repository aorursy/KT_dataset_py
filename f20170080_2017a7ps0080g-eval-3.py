from IPython.core.display import display, HTML

display(HTML("<style>.container { width:98% !important; }</style>"))
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



import os

from sklearn import preprocessing

import pickle

from sklearn.metrics import roc_auc_score
!ls ../input/eval-lab-3-f464
train_raw = pd.read_csv("../input/eval-lab-3-f464/train.csv")

test_raw = pd.read_csv("../input/eval-lab-3-f464/test.csv")
print(train_raw.shape)

train_raw.head()
print(test_raw.shape)

test_raw.head()
train_labels = train_raw['Satisfied']
labels, counts = np.unique(train_labels, return_counts=True)

print(counts)

plt.bar(labels, counts)

plt.show()
data_raw = pd.concat([train_raw.iloc[:,0:-1], test_raw])

print(data_raw.shape)
data_raw
for col in data_raw.columns:

    t = data_raw[col]

    print(col,":", t[t.isna()].shape[0])
for col in data_raw.columns:

    data_raw.loc[data_raw[col] == 'No tv connection',col] = 'No'

    data_raw.loc[data_raw[col] == 'No internet',col] = 'No'
from matplotlib import gridspec



fig = plt.figure(figsize=[30,10])

gs = gridspec.GridSpec(3, 6) 



for i,col in enumerate(data_raw.columns):

    if(col in ['custId', 'MonthlyCharges', 'TotalCharges']): continue



    t = data_raw[col]

    labels, counts = np.unique(t,return_counts=True)

    plt.subplot(gs[i-1])

    plt.title(col)

    plt.bar(labels, counts)

plt.show()

# isMale



gender = data_raw['gender']

isMale_idx = np.where(gender == 'Male')[0]

isMale = pd.DataFrame(np.zeros_like(gender), columns=['isMale'])

isMale.iloc[isMale_idx,0] = 1

isMale.head()
# isSenior



isSenior = pd.DataFrame(np.array(data_raw['SeniorCitizen']), columns=['isSenior'])

isSenior.head()
# isMarried



res = data_raw['Married']

isMarried_idx = np.where(res == 'Yes')[0]

isMarried = pd.DataFrame(np.zeros_like(res), columns=['isMarried'])

isMarried.iloc[isMarried_idx,0] = 1

isMarried.head()
# hasChildren



res = data_raw['Children']

hasChildren_idx = np.where(res == 'Yes')[0]

hasChildren = pd.DataFrame(np.zeros_like(res), columns=['hasChildren'])

hasChildren.iloc[hasChildren_idx,0] = 1

hasChildren.head()
# tvConnection



res = data_raw['TVConnection']

choices = ['No', 'Cable', 'DTH']

tvConnection =  pd.DataFrame(np.zeros([res.shape[0], 3]), columns=['hasConnection_'+choice for choice in choices])



for i,choice in enumerate(choices):

    choice_idx = np.where(res == choice)[0]

    tvConnection.iloc[choice_idx,i] = 1



tvConnection
# channels



channel_nos = ['Channel1', 'Channel2', 'Channel3', 'Channel4', 'Channel5', 'Channel6']

channels = pd.DataFrame(np.zeros([len(data_raw),6]), columns=[channel.lower() for channel in channel_nos])



for i,channel in enumerate(channel_nos):

    res = data_raw[channel]

    choice_idx = np.where(res == 'Yes')[0]

    channels.iloc[choice_idx,i] = 1

    

channels
# hasInternet



res = data_raw['Internet']

internet_idx = np.where(res == 'Yes')[0]

hasInternet = pd.DataFrame(np.zeros_like(res), columns=['hasInternet'])

hasInternet.iloc[internet_idx,0] = 1

hasInternet.head(10)
# isHighSpeed



res = data_raw['HighSpeed']

highSpeed_idx = np.where(res == 'Yes')[0]

isHighSpeed = pd.DataFrame(np.zeros_like(res), columns=['isHighSpeed'])

isHighSpeed.iloc[highSpeed_idx,0] = 1

isHighSpeed.head(10)
# addedServices



res = data_raw['AddedServices']

addedServices_idx = np.where(res == 'Yes')[0]

addedServices = pd.DataFrame(np.zeros_like(res), columns=['addedServices'])

addedServices.iloc[addedServices_idx,0] = 1

addedServices.head(10)
# subscription



res = data_raw['Subscription']

choices = ['Annually', 'Biannually', 'Monthly']

subscription =  pd.DataFrame(np.zeros([res.shape[0], 3]), columns=['billed'+choice for choice in choices])



for i,choice in enumerate(choices):

    choice_idx = np.where(res == choice)[0]

    subscription.iloc[choice_idx,i] = 1



subscription
# paymentMethod



res = data_raw['PaymentMethod']

choices = ['Bank transfer', 'Cash', 'Credit card', 'Net Banking']

paymentMethod =  pd.DataFrame(np.zeros([res.shape[0], 4]), columns=['pay'+choice.replace(' ','') for choice in choices])



for i,choice in enumerate(choices):

    choice_idx = np.where(res == choice)[0]

    paymentMethod.iloc[choice_idx,i] = 1



paymentMethod
dataframes = [

    'isMale',

    'isSenior',

    'isMarried',

    'hasChildren',

    'tvConnection',

    'channels',

    'hasInternet',

    'isHighSpeed',

    'addedServices',

    'subscription',

    'paymentMethod'

]



free = [

    'tenure', 

    'custId', 

    'MonthlyCharges', 

    'TotalCharges'

]



data = pd.concat([eval(dataframe) for dataframe in dataframes], sort=False, axis=1)

data2 = pd.concat([data_raw[free_idx] for free_idx in free], sort=False, axis=1).reset_index()

print(data.shape)

print(data2.shape)

data = pd.concat([data, data2], axis=1)

data = data.drop(['index'], axis=1)
for i,val in enumerate(data["TotalCharges"]):

    

    if(val != ' '):

        data.loc[i,"TotalCharges"] = float(val)

        

    else:

        data.loc[i,"TotalCharges"] = float('nan')
notna = data["TotalCharges"][data["TotalCharges"].notna()]

mean = notna.sum() / len(notna)

# print(mean)

data = data.fillna(mean)

# data[data['TotalCharges'].isna()]
data.columns
data = data.astype({

    'isMale' : np.int8,

    'isSenior' : np.int8,

    'isMarried' : np.int8,

    'hasChildren' : np.int8,

    'hasConnection_No' : np.int8,

    'hasConnection_Cable' : np.int8,

    'hasConnection_DTH' : np.int8,

    'channel1' : np.int8,

    'channel2' : np.int8,

    'channel3' : np.int8,

    'channel4' : np.int8,

    'channel5' : np.int8,

    'channel6' : np.int8,

    'hasInternet' : np.int8,

    'isHighSpeed' : np.int8,

    'addedServices' : np.int8,

    'billedAnnually' : np.int8,

    'billedBiannually' : np.int8,

    'billedMonthly' : np.int8,

    'payBanktransfer' : np.int8,

    'payCash' : np.int8,

    'payCreditcard' : np.int8,

    'payNetBanking' : np.int8,

    'tenure' : np.int8,

    'custId' : np.int16,

    'MonthlyCharges' : np.float64,

    'TotalCharges' : np.float64,

})
data.transpose()
data.dtypes
labels = train_raw['Satisfied']

labels = np.array(labels)
file = open('data.pkl', 'ab') 

pickle.dump(data, file)



file = open('labels.pkl', 'ab') 

pickle.dump(labels, file)       
file = open('data.pkl', 'rb')      

data = pickle.load(file) 



file = open('labels.pkl', 'rb')      

labels = pickle.load(file)



train_len = 4930

test_len = 2113



fdata = data
from sklearn.preprocessing import StandardScaler, RobustScaler

data = StandardScaler().fit_transform(fdata)

data = pd.DataFrame(data, columns=fdata.columns)

data.transpose()
cols = [

        'isMale', 

        'isSenior', 

        'isMarried', 

        'hasChildren', 

        'hasConnection_No',

        'hasConnection_Cable', 

        'hasConnection_DTH', 

        'channel1', 

        'channel2',

        'channel3', 

        'channel4', 

        'channel5', 

        'channel6', 

        'hasInternet',

        'isHighSpeed', 

        'addedServices', 

        'billedAnnually', 

        'billedBiannually',

        'billedMonthly', 

        'payBanktransfer', 

        'payCash', 

        'payCreditcard',

        'payNetBanking', 

        'custId', 

        'tenure', 

        'MonthlyCharges', 

        'TotalCharges'

       ]
new_col = [

#         'isMale', 

        'isSenior', 

#         'isMarried', 

#         'hasChildren', 

        'hasConnection_No',

        'hasConnection_Cable', 

#         'hasConnection_DTH', 

#         'channel1', 

#         'channel2',

#         'channel3', 

#         'channel4', 

        'channel5', 

        'channel6', 

#         'hasInternet',

#         'isHighSpeed', 

        'addedServices', 

#         'billedAnnually', 

#         'billedBiannually',

        'billedMonthly', 

#         'payBanktransfer', 

#         'payCash', 

#         'payCreditcard',

        'payNetBanking', 

#         'custId', 

#         'tenure', 

#         'MonthlyCharges', 

#         'TotalCharges'

       ]
lda_cols = [

#         'isSenior', 

#         'isMarried', 

#         'hasChildren', 

        'hasConnection_No',

        'hasConnection_Cable', 

#         'channel5', 

#         'channel6', 

        'addedServices', 

#         'billedAnnually', 

        'tenure', 

        'billedMonthly',

#         'TotalCharges'

       ]
insane_cols = [

        'billedMonthly'

       ]
non_corr = [

        'isMale', 

#         'isSenior', 

#         'isMarried', 

#         'hasChildren', 

#         'hasConnection_No',

#         'hasConnection_Cable', 

#         'hasConnection_DTH', 

#         'channel1', 

#         'channel2',

#         'channel3', 

#         'channel4', 

#         'channel5', 

#         'channel6', 

#         'hasInternet',

#         'isHighSpeed', 

#         'addedServices', 

#         'billedAnnually', 

#         'billedBiannually',

#         'billedMonthly', 

#         'payBanktransfer', 

#         'payCash', 

#         'payCreditcard',

#         'payNetBanking', 

#         'custId', 

        'tenure', 

        'MonthlyCharges', 

        'TotalCharges'

       ]
data = fdata[cols]
import seaborn as sns

check = pd.concat([data[:train_len], pd.DataFrame(labels, columns=['label'])], sort=False, axis=1)



plt.figure(figsize=(30,18))

sns.heatmap(check.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, cbar=False) 

plt.yticks(fontsize="20")

plt.xticks(fontsize="15", rotation=30)

plt.show()
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.decomposition import KernelPCA, PCA





def rand(data, pca_num=None):

#     data = PCA(pca_num).fit_transform(data) 

    X_train, X_val, y_train, y_val = train_test_split(

                                data[:train_len], 

                                labels, 

                                test_size=0.1, 

                                random_state=None, 

                        )

    

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)

#     X_train, y_train = ADASYN().fit_resample(X_train, y_train)

    return X_train, X_val, y_train, y_val



X_test = data[train_len:]
# %%time



import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss



skf = StratifiedKFold(n_splits=5)

train_data = data[:train_len]

# pca = PCA(n_components=8)

# train_data = pca.fit_transform(train_data)

# print(pca.explained_variance_ratio_)

# print("done")



train_avg = 0

val_avg = 0



train_data = np.asarray(train_data)

skf.get_n_splits(train_data, labels)



for train_idx,val_idx in skf.split(train_data, labels):



    X_train, y_train = train_data[train_idx,:], labels[train_idx]

    X_val, y_val = train_data[val_idx,:], labels[val_idx]

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)



    model = LinearDiscriminantAnalysis()

    t = model.fit(X_train, y_train) 



    train_score = roc_auc_score(y_train, model.predict(X_train))

    val_score = roc_auc_score(y_val, model.predict(X_val))

    train_avg += train_score

    val_avg += val_score



train_avg = train_avg / 5

val_avg = val_avg / 5



print(train_avg)

print(val_avg)

print(log_loss(y_train, model.predict_proba(X_train)))

print(log_loss(y_val, model.predict_proba(X_val)))
# %%time



import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix





skf = StratifiedKFold(n_splits=5)

train_data = data[:train_len]



lda = LinearDiscriminantAnalysis()

train_data = lda.fit_transform(data[:train_len], labels)

print(train_data.shape)



# pca = KernelPCA(n_components=8)

# train_data = pca.fit_transform(train_data)

# print(pca.explained_variance_ratio_)



# print("done")



train_avg = 0

val_avg = 0



train_data = np.asarray(train_data)

skf.get_n_splits(train_data, labels)



for train_idx,val_idx in skf.split(train_data, labels):



    X_train, y_train = train_data[train_idx,:], labels[train_idx]

    X_val, y_val = train_data[val_idx,:], labels[val_idx]

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)



    kmeans = KMeans(n_clusters=10, random_state=0)

    kmeans.fit(X_train)



    train_preds = kmeans.predict(X_train)



    cm = confusion_matrix(y_train,train_preds).astype(float)

    cm[0,] = cm[0,] / cm[0,].sum() * 100

    cm[1,] = cm[1,] / cm[1,].sum() * 100

    cluster_idx = np.argmax(cm, axis=0)



    train_preds = [cluster_idx[i] for i in train_preds]

    train_score = roc_auc_score(y_train, train_preds)



    val_preds = kmeans.predict(X_val)

    val_preds = [cluster_idx[i] for i in val_preds]

    val_score = roc_auc_score(y_val, val_preds)

    

    train_avg += train_score

    val_avg += val_score



train_avg = train_avg / 5

val_avg = val_avg / 5



print(train_avg)

print(val_avg)

# print(log_loss(y_train, model.predict_proba(X_train)))

# print(log_loss(y_val, model.predict_proba(X_val)))
# %%time



import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.cluster import KMeans

from sklearn.metrics import confusion_matrix



skf = StratifiedKFold(n_splits=5)

train_data = data[:train_len]



lda = LinearDiscriminantAnalysis()

train_data = lda.fit_transform(data[:train_len], labels)



train_avg = 0

val_avg = 0



train_data = np.asarray(train_data)

skf.get_n_splits(train_data, labels)



for train_idx,val_idx in skf.split(train_data, labels):



    X_train, y_train = train_data[train_idx,:], labels[train_idx]

    X_val, y_val = train_data[val_idx,:], labels[val_idx]

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)



    kmeans = KMeans(n_clusters=10, random_state=0)

    kmeans.fit(X_train)



    train_preds = kmeans.predict(X_train)



    cm = confusion_matrix(y_train,train_preds).astype(float)

    cm[0,] = cm[0,] / cm[0,].sum() * 100

    cm[1,] = cm[1,] / cm[1,].sum() * 100

    cluster_idx = np.argmax(cm, axis=0)



    train_preds = [cluster_idx[i] for i in train_preds]

    train_score = roc_auc_score(y_train, train_preds)



    val_preds = kmeans.predict(X_val)

    val_preds = [cluster_idx[i] for i in val_preds]

    val_score = roc_auc_score(y_val, val_preds)

    

    train_avg += train_score

    val_avg += val_score



train_avg = train_avg / 5

val_avg = val_avg / 5



print(train_avg)

print(val_avg)
%%time



import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.cluster import KMeans, AffinityPropagation

from sklearn.metrics import confusion_matrix



skf = StratifiedKFold(n_splits=3)

train_data = data[:train_len]



# lda = LinearDiscriminantAnalysis()

# train_data = lda.fit_transform(data[:train_len], labels)



pca = KernelPCA(n_components=5)

train_data = pca.fit_transform(data[:train_len], labels)



train_avg = 0

val_avg = 0



train_data = np.asarray(train_data)

skf.get_n_splits(train_data, labels)



for train_idx,val_idx in skf.split(train_data, labels):



    X_train, y_train = train_data[train_idx,:], labels[train_idx]

    X_val, y_val = train_data[val_idx,:], labels[val_idx]

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)



    kmeans = AffinityPropagation(verbose=True,convergence_iter=50, max_iter=300, damping=0.7)

    kmeans.fit(X_train)



    train_preds = kmeans.predict(X_train)



    cm = confusion_matrix(y_train,train_preds).astype(float)

    cm[0,] = cm[0,] / cm[0,].sum() * 100

    cm[1,] = cm[1,] / cm[1,].sum() * 100

    cluster_idx = np.argmax(cm, axis=0)



    train_preds = [cluster_idx[i] for i in train_preds]

    train_score = roc_auc_score(y_train, train_preds)



    val_preds = kmeans.predict(X_val)

    val_preds = [cluster_idx[i] for i in val_preds]

    val_score = roc_auc_score(y_val, val_preds)

    

    train_avg += train_score

    val_avg += val_score

    break



train_avg = train_avg / 1

val_avg = val_avg / 1



print(train_avg)

print(val_avg)
# %%time



import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from sklearn.cluster import KMeans, Birch

from sklearn.metrics import confusion_matrix



skf = StratifiedKFold(n_splits=5)

train_data = data[:train_len]



# lda = LinearDiscriminantAnalysis()

# train_data = lda.fit_transform(data[:train_len], labels)



train_avg = 0

val_avg = 0



train_data = np.asarray(train_data)

skf.get_n_splits(train_data, labels)



for train_idx,val_idx in skf.split(train_data, labels):



    X_train, y_train = train_data[train_idx,:], labels[train_idx]

    X_val, y_val = train_data[val_idx,:], labels[val_idx]

    X_train, y_train = SMOTE().fit_resample(X_train, y_train)



    kmeans = Birch(threshold=0.01)

    kmeans.fit(X_train)



    train_preds = kmeans.predict(X_train)



    cm = confusion_matrix(y_train,train_preds).astype(float)

    cm[0,] = cm[0,] / cm[0,].sum() * 100

    cm[1,] = cm[1,] / cm[1,].sum() * 100

    cluster_idx = np.argmax(cm, axis=0)



    train_preds = [cluster_idx[i] for i in train_preds]

    train_score = roc_auc_score(y_train, train_preds)



    val_preds = kmeans.predict(X_val)

    val_preds = [cluster_idx[i] for i in val_preds]

    val_score = roc_auc_score(y_val, val_preds)

    

    train_avg += train_score

    val_avg += val_score



train_avg = train_avg / 5

val_avg = val_avg / 5



print(train_avg)

print(val_avg)
# train_data = data[:train_len]



# # lda = LinearDiscriminantAnalysis()

# # train_data = lda.fit_transform(data[:train_len], labels)



# pca = KernelPCA(n_components=5)

# train_data = pca.fit(data[:train_len], labels)

# test_data = pca.transform(data[:test_len], labels)



# X_train, y_train = SMOTE().fit_resample(X_train, y_train)



# kmeans = AffinityPropagation(verbose=True,convergence_iter=50, max_iter=300, damping=0.7)

# kmeans.fit(X_train)



# train_preds = kmeans.predict(X_train)



# cm = confusion_matrix(y_train,train_preds).astype(float)

# cm[0,] = cm[0,] / cm[0,].sum() * 100

# cm[1,] = cm[1,] / cm[1,].sum() * 100

# cluster_idx = np.argmax(cm, axis=0)



# train_preds = [cluster_idx[i] for i in train_preds]

# train_score = roc_auc_score(y_train, train_preds)



# val_preds = kmeans.predict(X_val)

# val_preds = [cluster_idx[i] for i in val_preds]

# val_score = roc_auc_score(y_val, val_preds)
pca_num = None

# data = PCA(pca_num).fit_transform(data)



# pca = KernelPCA(n_components=8)

# data = pca.fit_transform(data)



X_train = data[:train_len]

X_test = data[train_len:]

y_train = labels

X_train, y_train = SMOTE().fit_resample(X_train, y_train)



model = LinearDiscriminantAnalysis()

t = model.fit(X_train, y_train)

roc_auc_score(y_train, model.predict(X_train))
def generate_submission(model, fname):

    pred_test = model.predict(X_test)



    sub = pd.DataFrame({

        'custId' : test_raw['custId'],

        'Satisfied': pred_test

    })

    

    sub.to_csv(fname, index=False)

    return sub
# generate_submission(model, 'pred21-all_features-lda-no_pca-SMOTE-full_data-st_normalized.csv')

generate_submission(model, 'predxx-Final.csv')

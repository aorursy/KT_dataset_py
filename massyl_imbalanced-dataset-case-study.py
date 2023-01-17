# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel, RFE
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')

data.info()
data.describe()
sns.countplot('Class', data=data)
data['Class'].value_counts()
fig, ax = plt.subplots (5, 6, figsize=(20,15))

for i, ax in enumerate(ax.flatten()):
    sns.distplot(data[data['Class']==0] [data.columns[i]], ax= ax)
    sns.distplot(data[data['Class']==1] [data.columns[i]], ax= ax)

plt.tight_layout()
data ['Amount'] = StandardScaler().fit_transform(np.log(data['Amount'].values + 1e-3).reshape(-1, 1))
fig, ax = plt.subplots (5, 6, figsize=(20,15))

for i, ax in enumerate(ax.flatten()):
    sns.boxplot('Class', data.columns[i], data=data, ax= ax)

plt.tight_layout()
plt.show()
plt.figure(figsize=(8,8))
sns.heatmap(data.corr(), cmap='coolwarm')
subsampled_data = pd.concat([data[data['Class']==0] .sample(frac=.05), data[data['Class']==1]]).sample(frac=1)
fig, ax = plt.subplots (5, 6, figsize=(15,20))

for i, ax in enumerate(ax.flatten()):
    sns.distplot(subsampled_data[subsampled_data['Class']==0] [subsampled_data.columns[i]], ax= ax)
    sns.distplot(subsampled_data[subsampled_data['Class']==1] [subsampled_data.columns[i]], ax= ax)

plt.tight_layout()
data ['Amount'] = StandardScaler().fit_transform(np.log(data['Amount'].values + 1e-3).reshape(-1, 1))
params =['V1', 'V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
def pca (df, params, n_cp=2):
    
    X = df[params].values

    pca = PCA(n_components=n_cp)
    pca.fit(X)

    X_transformed= pca.transform(X)


    data_pca = df[['Class']].copy()
    if n_cp==2:
        data_pca[['PC1', 'PC2']]= X_transformed
    else:
        data_pca[['PC1', 'PC2', 'PC3', 'PC4']]= X_transformed
        
    print(pca.explained_variance_ratio_)
    return data_pca
data_pca_all_features = pca (subsampled_data, subsampled_data.columns)
sns.scatterplot('PC1', 'PC2', hue='Class', data=data_pca_all_features)
data_pca_manual_selected_feat = pca (subsampled_data, params)
sns.scatterplot('PC1', 'PC2', hue='Class', data=data_pca_manual_selected_feat)
n = 10

X = subsampled_data[subsampled_data.columns[:-1]] .values
y = subsampled_data[subsampled_data.columns[-1]] .values
sel = SelectKBest(k=n).fit(X, y)

cols = subsampled_data[subsampled_data.columns[:-1]].columns[sel.get_support()].tolist()

data_pca_selectKBest = pca (subsampled_data, cols)
sns.scatterplot('PC1', 'PC2', hue='Class', data=data_pca_selectKBest)
X = subsampled_data[subsampled_data.columns[:-1]] .values
y = subsampled_data[subsampled_data.columns[-1]] .values

lsvc = svm.LinearSVC(C=0.001, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)

cols = subsampled_data[subsampled_data.columns[:-1]].columns[model.get_support()].tolist()

data_pca_lasso_feat = pca (subsampled_data, cols)
sns.scatterplot('PC1', 'PC2', hue='Class', data=data_pca_lasso_feat)
print(len(model.get_support()[model.get_support()==True]), len(data.columns))
n = 10

X = subsampled_data[subsampled_data.columns[:-1]] .values
y = subsampled_data[subsampled_data.columns[-1]] .values
sel = SelectKBest(k=n).fit(X, y)

cols = subsampled_data[subsampled_data.columns[:-1]].columns[sel.get_support()].tolist()

X = subsampled_data[cols] .values

X_transformed= TSNE(n_components=2).fit_transform(X)

data_tsne = subsampled_data[['Class']].copy()
data_tsne[['PC1', 'PC2']]= X_transformed

sns.scatterplot('PC1', 'PC2', hue='Class', data=data_tsne)
df = pd.concat([subsampled_data[subsampled_data['Class']==0] .sample(frac=.05), subsampled_data[subsampled_data['Class']==1]]).sample(frac=1)

X = df[df.columns[:-1]] .values
y = df[df.columns[-1]] .values

svc = svm.SVC(kernel="linear")
rfecv = RFE(estimator=svc)
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

cols = subsampled_data[subsampled_data.columns[:-1]].columns[rfecv.get_support()].tolist()

data_pca_rfecv_feat = pca (subsampled_data, cols)
sns.scatterplot('PC1', 'PC2', hue='Class', data=data_pca_rfecv_feat)

print(len(model.get_support()[rfecv.get_support()==True]), len(data.columns))
titles = ["all_features", "lasso_feat", "manual_selected_feat", "rfecv_feat", "selectKBest", "tsne"]
datasets = [data_pca_all_features, data_pca_lasso_feat, data_pca_manual_selected_feat, data_pca_rfecv_feat, data_pca_selectKBest, data_tsne]

fig, ax = plt.subplots (3,2, figsize=(15,8))

for i, ax in enumerate(ax.flatten()):
    X = datasets[i][['PC1', 'PC2']].values
    y = datasets[i]['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    clf = svm.SVC(random_state=0)
    clf.fit(X_train, y_train)

    plot_confusion_matrix(clf, X_test, y_test, ax=ax, normalize ='true')
    ax.set_title(titles[i])
    
plt.tight_layout()
plt.show()
data_pca_manual_selected_feat = pca (subsampled_data, params, 4)

cols = subsampled_data[subsampled_data.columns[:-1]].columns[rfecv.get_support()].tolist()
data_pca_rfecv_feat = pca (subsampled_data, cols, 4)

cols = subsampled_data[subsampled_data.columns[:-1]].columns[sel.get_support()].tolist()
data_pca_selectKBest = pca (subsampled_data, cols, 4)

df = pd.concat([subsampled_data[subsampled_data['Class']==0] .sample(frac=.05), subsampled_data[subsampled_data['Class']==1]]).sample(frac=1)
cols = df[df.columns[:-1]].columns[sel.get_support()].tolist()
X = df[cols] .values
X_transformed= TSNE(n_components=3).fit_transform(X)
data_tsne = df[['Class']].copy()
data_tsne[['PC1', 'PC2', 'PC3']]= X_transformed
titles = ["manual_selected_feat", "rfecv_feat", "selectKBest", "tsne"]
datasets = [data_pca_manual_selected_feat, data_pca_rfecv_feat, data_pca_selectKBest, data_tsne]

fig, ax = plt.subplots (2,2, figsize=(15,8))

for i, ax in enumerate(ax.flatten()):
    X = datasets[i][['PC1', 'PC2', 'PC3']].values
    y = datasets[i]['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    clf = svm.SVC(random_state=0)
    clf.fit(X_train, y_train)

    plot_confusion_matrix(clf, X_test, y_test, ax=ax, normalize ='true')
    ax.set_title(titles[i])
    
plt.tight_layout()
plt.show()
scaler = StandardScaler()
X_train = data_pca_rfecv_feat[data_pca_rfecv_feat['Class']==0] [['PC1', 'PC2', 'PC3']].values

ratio =  len(data_pca_rfecv_feat[data_pca_rfecv_feat['Class']==1] )/ len(data_pca_rfecv_feat)
print(ratio)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(data_pca_rfecv_feat[['PC1', 'PC2', 'PC3']].values)
y = data_pca_rfecv_feat['Class']

# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
clf = svm.OneClassSVM(nu=0.1, gamma='auto').fit(X_train)
y_pred = [1 if i==-1 else 0 for i in clf.predict(X_test) ]

print(confusion_matrix(y, y_pred))



clf = IsolationForest(n_estimators=200, contamination=ratio).fit(X_train)
y_pred = [1 if i==-1 else 0 for i in clf.predict(X_test) ]

print(confusion_matrix(y, y_pred))



clf = EllipticEnvelope(contamination=ratio).fit(X_train)
y_pred = [1 if i==-1 else 0 for i in clf.predict(X_test) ]

print(confusion_matrix(y, y_pred))
clf = LocalOutlierFactor(n_neighbors=2) 
y_pred = [1 if i==-1 else 0 for i in clf.fit_predict(X_test) ]

print(confusion_matrix(y, y_pred))



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

    
def train_nn(X_train, X_test, y_train, y_test, verbose=False):
    model = keras.Sequential() 

    model.add(layers.Dense(20, activation="relu", input_dim=X_train.shape[1]))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(10, activation="relu",))
    model.add(layers.Dropout(.2))
    model.add(layers.Dense(10, activation="relu",))
    model.add(layers.Dense(1, activation='sigmoid'))


    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])

    train_history = model.fit(X_train, y_train, epochs=100, batch_size=50, verbose=0,validation_split=0.8,) 


    scores = model.evaluate(X_test, y_test)
    print('\n')
    print('accuracy=',scores[1])

    y_pred = model.predict_classes(X_test)

    print (titles[i])
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    if verbose:
        show_train_history(train_history,'accuracy','val_accuracy')
        show_train_history(train_history,'loss','val_loss')
        
        print('Classification Report')
        print(classification_report(y_test, y_pred))
subsampled_data_nn = pd.concat([data[data['Class']==0] .sample(frac=.01), data[data['Class']==1]]).sample(frac=1)
scaler = MinMaxScaler()

X = subsampled_data_nn.drop('Class',1).values
y = subsampled_data_nn['Class'].values.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)

scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

train_nn(X_train, X_test, y_train, y_test, True)
titles = ["manual_selected_feat", "rfecv_feat", "selectKBest", "tsne"]
datasets = [data_pca_manual_selected_feat, data_pca_rfecv_feat, data_pca_selectKBest, data_tsne]

for i in range(len(datasets)):
    X = datasets[i][['PC1', 'PC2', 'PC3']].values
    y = datasets[i]['Class'].values.astype('float32')

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    train_nn(X_train, X_test, y_train, y_test)
X = subsampled_data[subsampled_data['Class']==0] [subsampled_data.columns[:-1]].values
X = MinMaxScaler().fit_transform(X)
encoder = keras.Sequential()
encoder.add(layers.Input(30))
encoder.add(layers.Dense(100, activation="relu"))
encoder.add(layers.Dense(50, activation='relu'))
encoder.add(layers.Dense(100, activation='relu'))
encoder.add(layers.Dense(30, ))

encoder.compile(optimizer=keras.optimizers.Adam(learning_rate=3e-4),loss='mse')

train_history=encoder.fit(X, X, epochs=50, verbose=0, validation_split=0.8)

show_train_history(train_history,'loss','val_loss')
hidden_representation = keras.Sequential()
hidden_representation.add(encoder.layers[0])
hidden_representation.add(encoder.layers[1])


X_non_fraud = subsampled_data[subsampled_data['Class']==0] [subsampled_data.columns[:-1]].values
X_fraud = subsampled_data[subsampled_data['Class']==1] [subsampled_data.columns[:-1]].values


X_non_fraud = MinMaxScaler().fit_transform(X_non_fraud)
X_fraud = MinMaxScaler().fit_transform(X_fraud)


non_fraud_hid_rep = hidden_representation.predict(X_non_fraud)
fraud_hid_rep = hidden_representation.predict(X_fraud)
rep_x = np.append(non_fraud_hid_rep, fraud_hid_rep, axis = 0)
y_n = np.zeros(non_fraud_hid_rep.shape[0])
y_f = np.ones(fraud_hid_rep.shape[0])
rep_y = np.append(y_n, y_f)

X_transformed= TSNE(n_components=2).fit_transform(rep_x)

sns.scatterplot(X_transformed[:,0] , X_transformed[:,1], hue=rep_y)
X = X_transformed
y = rep_y

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y, shuffle =True)
clf = svm.SVC(C = 1,random_state=0)
clf.fit(X_train, y_train)

plot_confusion_matrix(clf, X_test, y_test, normalize ='true')
h = 1

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
x_min, x_max = X_transformed[:, 0].min() - 1, X_transformed[:, 0].max() + 1
y_min, y_max = X_transformed[:, 1].min() - 1, X_transformed[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

color_map = {-1: (1, 1, 1), 0: (0, 0, .9), 1: (1, 0, 0), 2: (.8, .6, 0)}
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('off')

# Plot also the training points
colors = [color_map[y] for y in y]
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=colors, edgecolors='black')
subsampled_data = pd.concat([data[data['Class']==0] .sample(frac=.05), data[data['Class']==1]]).sample(frac=1)
from imblearn.over_sampling import SMOTE
from collections import Counter

X = subsampled_data[subsampled_data.columns[:-1]] .values
y = subsampled_data[subsampled_data.columns[-1]] .values

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)


smote_data = pd.DataFrame(data=X_res, columns=subsampled_data.columns[:-1])
smote_data['Class'] = y_res

len(smote_data)
fig, ax = plt.subplots (5, 6, figsize=(20,15))

for i, ax in enumerate(ax.flatten()):
    sns.distplot(smote_data[smote_data['Class']==0] [smote_data.columns[i]], ax= ax)
    sns.distplot(smote_data[smote_data['Class']==1] [smote_data.columns[i]], ax= ax)

plt.tight_layout()
data_pca_manual_selected_feat = pca (smote_data, params, 2)

cols = smote_data[smote_data.columns[:-1]].columns[rfecv.get_support()].tolist()
data_pca_rfecv_feat = pca (smote_data, cols, 2)

cols = smote_data[smote_data.columns[:-1]].columns[sel.get_support()].tolist()
data_pca_selectKBest = pca (smote_data, cols, 2)

cols = smote_data[smote_data.columns[:-1]].columns[sel.get_support()].tolist()
X = smote_data[cols] .values
X_transformed= TSNE(n_components=2).fit_transform(X)
data_tsne = smote_data[['Class']].copy()
data_tsne[['PC1', 'PC2']]= X_transformed
titles = ["manual_selected_feat", "rfecv_feat", "selectKBest", "tsne"]
datasets = [data_pca_manual_selected_feat, data_pca_rfecv_feat, data_pca_selectKBest, data_tsne]

fig, ax = plt.subplots (2,2, figsize=(15,8))

for i, ax in enumerate(ax.flatten()):
    
    sns.scatterplot('PC1', 'PC2', hue="Class", data=datasets[i], ax=ax)
    ax.set_title(titles[i])
    
plt.tight_layout()
plt.show()
titles = ["manual_selected_feat", "rfecv_feat", "selectKBest", "tsne"]
datasets = [data_pca_manual_selected_feat, data_pca_rfecv_feat, data_pca_selectKBest, data_tsne]

fig, ax = plt.subplots (2,2, figsize=(15,8))

for i, ax in enumerate(ax.flatten()):
    X = datasets[i][['PC1', 'PC2']].values
    y = datasets[i]['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    clf = svm.SVC(random_state=0)
    clf.fit(X_train, y_train)

    plot_confusion_matrix(clf, X_test, y_test, ax=ax, normalize ='true')
    ax.set_title(titles[i])
    
plt.tight_layout()
plt.show()
from imblearn.under_sampling import NearMiss

X = data[data.columns[:-1]] .values
y = data[data.columns[-1]] .values

nm = NearMiss(version=2)
X_res, y_res = nm.fit_resample(X, y)


nearMiss_data = pd.DataFrame(data=X_res, columns=data.columns[:-1])
nearMiss_data['Class'] = y_res

len(nearMiss_data)
fig, ax = plt.subplots (5, 6, figsize=(20,15))

for i, ax in enumerate(ax.flatten()):
    sns.distplot(nearMiss_data[nearMiss_data['Class']==0] [nearMiss_data.columns[i]], ax= ax)
    sns.distplot(nearMiss_data[nearMiss_data['Class']==1] [nearMiss_data.columns[i]], ax= ax)

plt.tight_layout()
data_pca_manual_selected_feat = pca (nearMiss_data, params, 2)

cols = nearMiss_data[nearMiss_data.columns[:-1]].columns[rfecv.get_support()].tolist()
data_pca_rfecv_feat = pca (nearMiss_data, cols, 2)

cols = nearMiss_data[nearMiss_data.columns[:-1]].columns[sel.get_support()].tolist()
data_pca_selectKBest = pca (nearMiss_data, cols, 2)

cols = nearMiss_data[nearMiss_data.columns[:-1]].columns[sel.get_support()].tolist()
X = nearMiss_data[cols] .values
X_transformed= TSNE(n_components=2).fit_transform(X)
data_tsne = nearMiss_data[['Class']].copy()
data_tsne[['PC1', 'PC2']]= X_transformed
titles = ["manual_selected_feat", "rfecv_feat", "selectKBest", "tsne"]
datasets = [data_pca_manual_selected_feat, data_pca_rfecv_feat, data_pca_selectKBest, data_tsne]

fig, ax = plt.subplots (2,2, figsize=(15,8))

for i, ax in enumerate(ax.flatten()):
    
    sns.scatterplot('PC1', 'PC2', hue="Class", data=datasets[i], ax=ax)
    ax.set_title(titles[i])
    
plt.tight_layout()
plt.show()
titles = ["manual_selected_feat", "rfecv_feat", "selectKBest", "tsne"]
datasets = [data_pca_manual_selected_feat, data_pca_rfecv_feat, data_pca_selectKBest, data_tsne]

fig, ax = plt.subplots (2,2, figsize=(15,8))

for i, ax in enumerate(ax.flatten()):
    X = datasets[i][['PC1', 'PC2']].values
    y = datasets[i]['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    clf = svm.SVC(random_state=0)
    clf.fit(X_train, y_train)

    plot_confusion_matrix(clf, X_test, y_test, ax=ax, normalize ='true')
    ax.set_title(titles[i])
    
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

%matplotlib inline

from collections import Counter,defaultdict

from itertools import combinations 

import pandas_profiling

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import numpy as np

import os

import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler,LabelEncoder

%matplotlib inline

from sklearn.metrics import classification_report, confusion_matrix 

from IPython.display import HTML

import base64

from sklearn.decomposition import PCA

import operator

from sklearn.cluster import DBSCAN,KMeans,Birch,SpectralClustering,AgglomerativeClustering,MeanShift

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis

from sklearn.metrics import accuracy_score,f1_score

from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,BaggingClassifier,VotingClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.model_selection import cross_val_score, train_test_split,StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

from skimage.feature import canny

import xgboost as xgb

from sklearn.svm import SVC
!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!ls ~/.kaggle
!pip install -q kaggle

!pip install -q kaggle-cli
!kaggle competitions download -c eval-lab-4-f464
!unzip train.npy.zip

!unzip test.npy.zip
train_x = np.load("train.npy",allow_pickle=True)

test_x = np.load("test.npy",allow_pickle=True)
def rgb2gray(rgb):

    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def create_download_link(df, title = "Download CSV file",count=[0]):

    count[0] = count[0]+1

    filename = "data"+str(count[0])+".csv"

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

labels = []

for i in range(train_x.shape[0]):

    labels.append(train_x[i][0])

print(Counter(labels))
d=Counter(labels)

label_conv={}

for i in range(len(d)):

    label_conv[list(d.keys())[i]]=i

label_conv
rev_conv_label = {}

for i,j in label_conv.items():

    rev_conv_label[j]=i

print(rev_conv_label)
trainimages = np.array([j for i,j in train_x])

traingreyscale = np.array([rgb2gray(i) for i in trainimages])

trainxrgb = np.array([np.reshape(i,(7500)) for i in trainimages])

trainxgrey = np.array([np.reshape(i,(2500)) for i in traingreyscale])

trainxcanny =np.array([canny(i,sigma=1.0) for i in traingreyscale])

trainxcannyflat = np.array([np.reshape(i,(2500)) for i in trainxcanny])

trainy = np.array([label_conv[i] for i,j in train_x])
plt.imshow(traingreyscale[3],cmap='gray', vmin=0, vmax=255)

plt.show()

plt.imshow(trainxcanny[3],cmap='gray', vmin=0, vmax=1)

plt.show()

print(rev_conv_label[trainy[3]])
lda = LinearDiscriminantAnalysis(n_components=18)

lda.fit(trainxrgb,trainy)

trainxlda18rgb = lda.transform(trainxrgb)
trainxavgrgb = np.array([np.mean(image, axis=(0, 1)) for image in trainimages]) #to check average face colour
pca=PCA()

pca.fit(trainxrgb)



plt.figure(1, figsize=(12,8))



plt.plot(pca.explained_variance_, linewidth=2)

 

plt.xlabel('Components')

plt.ylabel('Explained Variaces')

plt.show()

pca=PCA()

pca.fit(trainxcannyflat)

plt.figure(1, figsize=(12,8))

plt.plot(pca.explained_variance_, linewidth=2) 

plt.xlabel('Components')

plt.ylabel('Explained Variaces')

plt.show()
n_components = 250

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(trainxrgb)

trainxpca = pca.transform(trainxrgb)
n_components = 500

pca1 = PCA(n_components=n_components, svd_solver='randomized', whiten=True, random_state=42).fit(trainxcannyflat)

trainxcannypca = pca1.transform(trainxcannyflat)
model_a = KMeans(n_clusters=50)

model_a.fit(trainxlda18rgb)

trainxkmeansclusteridoflda = model_a.predict(trainxlda18rgb)

print(Counter(trainxkmeansclusteridoflda))
model_b = KMeans(n_clusters=50)

model_b.fit(trainxrgb)

trainxkmeansclusterid = model_b.predict(trainxrgb)
model_c = KMeans(n_clusters=50)

model_c.fit(trainxcannypca)

trainxkmeansclusteridofpca = model_c.predict(trainxcannypca)
total_viable_features = ['pcaimg','ldargb','avgrgb','cannypca','cannyflat','clusteridpca','clusteridrgb','clusteridlda']
def get_data(label):

    data_dict = {"pcaimg":trainxpca,

                 "ldargb":trainxlda18rgb,

                 "avgrgb":trainxavgrgb,

                 "cannypca":trainxcannypca,

                 "cannyflat":trainxcannyflat,

                 "clusteridpca":np.reshape(trainxkmeansclusteridofpca,(2275,1)),

                 "clusteridrgb":np.reshape(trainxkmeansclusterid,(2275,1)),

                 "clusteridlda":np.reshape(trainxkmeansclusteridoflda,(2275,1))}

    return data_dict[label]
best_model = None

best_features = None

best_score = 0

max_features = 5 #choose higher value for better results max 8

s=[]
# for n_features in range(1,max_features):

#     for features in combinations(total_viable_features,n_features):

#         n = n_features

#         data = [get_data(i) for i in features]

#         traindata = np.concatenate([get_data(i) for i in features],axis=1) #merging the different features

#         print("Calculating for"+str(features))

#         svc = SVC(kernel='rbf', class_weight='balanced')

#         param_grid = {'C': [1,5,10,50],

#               'gamma': [0.0001,0.001,0.005,0.01]}

#         grid = GridSearchCV(svc, param_grid, cv=4,verbose=10,scoring='f1_micro')

#         grid.fit(traindata, trainy)

#         print("Best params are ",grid.best_params_,sep=' ')

#         if grid.best_score_ >best_score:

#             best_model = grid.best_estimator_ 

#             best_features = features

#             best_score = grid.best_score_

#             best_params = grid.best_params_

       

# for n_features in range(1,max_features):

#     for features in combinations(total_viable_features,n_features):

#         n = n_features

#         data = [get_data(i) for i in features]

#         traindata = np.concatenate([get_data(i) for i in features],axis=1) #merging the different features

#         print("Calculating for"+str(features))

#         rfc=RandomForestClassifier(random_state=42)

#         param_grid = { 

#                         'n_estimators': [100, 500,1200],

#                         'max_features': ['auto', 'log2'],

#                         'max_depth' : [None,5,8,10],

#                         'criterion' :['gini', 'entropy']

#                     }

#         grid = GridSearchCV(rfc, param_grid, cv=4,verbose=10,scoring='f1_micro')

#         grid.fit(traindata, trainy)

#         print("Best params are ",grid.best_params_,sep=' ')

#         if grid.best_score_ >best_score:

#             best_model = grid.best_estimator_ 

#             best_features = features

#             best_score = grid.best_score_

#             best_params = grid.best_params_
# for n_features in range(1,max_features):

#     for features in combinations(total_viable_features,n_features):

#         n = n_features

#         data = [get_data(i) for i in features]

#         traindata = np.concatenate([get_data(i) for i in features],axis=1) #merging the different features

#         print("Calculating for"+str(features))

#         clf = xgb.XGBClassifier(random_state=42)

#         param_grid = { 

#                         'n_estimators': [100,500,1000],

#                         'learning_rate ': [0.1,0.01],

#                         'max_depth' : [3,5,8],

#                         'gamma' :[0,0.1]

#                     }

#         grid = GridSearchCV(clf, param_grid, cv=4,verbose=10,scoring='f1_micro')

#         grid.fit(traindata, trainy)

#         print("Best params are ",grid.best_params_,sep=' ')

#         if grid.best_score_ >best_score:

#             best_model = grid.best_estimator_ 

#             best_features = features

#             best_score = grid.best_score_

#             best_params = grid.best_params_
# for n_features in range(1,max_features):

#     for features in combinations(total_viable_features,n_features):

#         n = n_features

#         data = [get_data(i) for i in features]

#         traindata = np.concatenate([get_data(i) for i in features],axis=1) #merging the different features

#         print("Calculating for"+str(features))

#         clf = ExtraTreesClassifier(random_state=42)

#         param_grid = { 

#                         'n_estimators': [100, 500,1200],

#                         'max_features': ['auto', 'log2'],

#                         'max_depth' : [None,5,8,10],

#                         'criterion' :['gini', 'entropy'],

#                         'bootstrap' :['True','False']

#                     }

#         grid = GridSearchCV(clf, param_grid, cv=4,verbose=10,scoring='f1_micro')

#         grid.fit(traindata, trainy)

#         print("Best params are ",grid.best_params_,sep=' ')

#         if grid.best_score_ >best_score:

#             best_model = grid.best_estimator_ 

#             best_features = features

#             best_score = grid.best_score_

#             best_params = grid.best_params_
model = best_model
test_images = np.array([j for i,j in test_x])

testgreyscale = np.array([rgb2gray(i) for i in test_images])

testxrgb = np.array([np.reshape(i,(7500)) for i in test_images])

testxgrey = np.array([np.reshape(i,(2500)) for i in testgreyscale])

testxlda  = lda.transform(testxrgb)

testxavgrgb = np.array([np.mean(image, axis=(0, 1)) for image in test_images])

testxcanny =np.array([canny(i,sigma=1.0) for i in testgreyscale])

testxcannyflat = np.array([np.reshape(i,(2500)) for i in testxcanny])

testxcannypca = pca1.transform(testxcannyflat)

testxpca = pca.transform(testxrgb)

testxkmeansclusteridoflda = model_a.predict(testxlda)

testxkmeansclusterid = model_b.predict(testxrgb)

testxkmeansclusteridofpca = model_c.predict(testxcannypca)

test_id = np.array([i for i,j in test_x])
def get_data_test(label):

    data_dict = {"pcaimg":testxpca,

                 "ldargb":testxlda,

                 "avgrgb":testxavgrgb,

                 "cannypca":testxcannypca,

                 "cannyflat":testxcannyflat,

                 "clusteridpca":np.reshape(testxkmeansclusteridofpca,(976,1)),

                 "clusteridrgb":np.reshape(testxkmeansclusterid,(976,1)),

                 "clusteridlda":np.reshape(testxkmeansclusteridoflda,(976,1))}

    return data_dict[label]
test_new = np.concatenate([get_data_test(i) for i in best_features],axis=1)
pred = model.predict(test_new)
df = pd.read_csv("sample_submission.csv")

df["Celebrity"] = df["ImageId"].apply( lambda x: rev_conv_label[pred[int(x)]])
create_download_link(df)
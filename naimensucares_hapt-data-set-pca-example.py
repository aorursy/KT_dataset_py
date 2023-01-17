# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import random 
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import matplotlib.pyplot as plt  
import warnings            
warnings.filterwarnings("ignore") 

file=open("/kaggle/input/simplified-human-activity-recognition-wsmartphone/features_info.txt")
print(file.read())
file.close()
file=open("/kaggle/input/simplified-human-activity-recognition-wsmartphone/activity_labels.txt")
print(file.read())
file.close()

x_train=pd.read_csv("/kaggle/input/simplified-human-activity-recognition-wsmartphone/train_X.csv")
x_train=x_train.iloc[:,1:]
y_train=pd.read_csv("/kaggle/input/simplified-human-activity-recognition-wsmartphone/train_Y.csv")
y_train=y_train.iloc[:,1:]
x_test=pd.read_csv("/kaggle/input/simplified-human-activity-recognition-wsmartphone/test_X.csv")
x_test=x_test.iloc[:,1:]
y_test=pd.read_csv("/kaggle/input/simplified-human-activity-recognition-wsmartphone/test_Y.csv")
y_test=y_test.iloc[:,1:]


display(x_train.info())
display(x_train)

display(y_train.info())
display(y_train)

display(x_test.info())
display(x_test)

display(y_test.info())
display(y_test)
print(x_train.isnull().sum().sum())
print(x_test.isnull().sum().sum())
print(x_train.isnull().sum().sum())
print(y_test.isnull().sum().sum())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

knn = KNeighborsClassifier()
knn_model = knn.fit(x_train, y_train)
knn_model

y_pred = knn_model.predict(x_test)
knn_x_train_score=accuracy_score(y_test, y_pred)
knn_x_train_score
print(classification_report(y_test, y_pred))
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#scale etmemiz gerekiyor 
df = StandardScaler().fit_transform(x_train)

pca = PCA()
X_pca_train = pca.fit_transform(df)

#bu fonksiyon modelimizdeki degiskenleri temsil etme derecesini veriyor.
pca.explained_variance_ratio_[:10]
#daha anlasilir olsun diye bu sekilde yazildi % olarak temsil etme derecesi
# 64.degisken ve ustu degerler %90 dan fazla temsil etmektedir 

represent=np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)
print(represent)

import matplotlib.pyplot as plt

plt.scatter(X_pca_train[:,0],X_pca_train[:,1],color = ["red"],s=80)
plt.scatter(X_pca_train[:,0],X_pca_train[:,3],color = ["black"],s=70)
plt.scatter(X_pca_train[:,0],X_pca_train[:,7],color = ["orange"],s=60)
plt.scatter(X_pca_train[:,0],X_pca_train[:,15],color = ["purple"],s=40)
plt.scatter(X_pca_train[:,0],X_pca_train[:,20],color = ["green"],s=20)
plt.scatter(X_pca_train[:,0],X_pca_train[:,500],color = ["yellow"],s=10)

plt.legend()
plt.xlabel("x_train_0")
plt.ylabel("x_train_(1-8)")

plt.show()
X_pca_train
df_test = StandardScaler().fit_transform(x_test)

pca_test = PCA()
X_pca_test = pca_test.fit_transform(df_test)


represent=np.cumsum(np.round(pca_test.explained_variance_ratio_, decimals = 4)*100)
print(represent)

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np


knn_pca = KNeighborsClassifier()
knn_pca.fit(X_pca_train, y_train)
y_pred = knn_pca.predict(X_pca_test)
print(accuracy_score(y_test, y_pred))
knn_pca_score=cross_val_score(knn_pca, X_pca_test, y_test, cv = 10).mean()
knn_pca_score
knn_pca_rep = KNeighborsClassifier()

from sklearn import model_selection
cv_10 = model_selection.KFold(n_splits = 10,
                             shuffle = True,
                             random_state = 1)

features = []
# %90 ayarlandi
for i in np.arange(1, 30):
   
    
    score =model_selection.cross_val_score(knn_pca_rep,
                                           X_pca_train[:,:i], 
                                           y_train,
                                           cv=cv_10  )
    features.append(score)
     
plt.plot(features,'-v')
plt.xlabel('Bileşen Sayısı')
plt.ylabel('% ')
plt.title(' PCR Model Tuning');
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(X_pca_train[:,0],X_pca_train[:,1],s=100 )
ax.scatter(X_pca_train[:,0],X_pca_train[:,2],  s=60),
ax.scatter(X_pca_train[:,0],X_pca_train[:,3],  s=30),
ax.scatter(X_pca_train[:,0],X_pca_train[:,4],  s=20),
ax.scatter(X_pca_train[:,0],X_pca_train[:,5],  s=12),
ax.scatter(X_pca_train[:,0],X_pca_train[:,10],  s=8),
ax.scatter(X_pca_train[:,0],X_pca_train[:,15],  s=5),
ax.scatter(X_pca_train[:,0],X_pca_train[:,25],  s=4);
knn_pca_rep.fit(X_pca_train[:,:25], y_train)

y_pred = knn_pca_rep.predict(X_pca_test[:,:25])
print(accuracy_score(y_test, y_pred))
knn_pca_rep_score=cross_val_score(knn_pca_rep, X_pca_test[:,:25], y_test, cv = 10).mean()
knn_pca_rep_score

from sklearn.naive_bayes import GaussianNB
nb_x_train = GaussianNB()
nb_model_x_train = nb_x_train.fit(x_train, y_train)
y_pred=nb_model_x_train.predict(x_test)
print(accuracy_score(y_test, y_pred))
NB_x_train_score=cross_val_score(nb_model_x_train, x_test, y_test, cv = 10).mean()
NB_x_train_score

nb_pca = GaussianNB()
nb_pca_model = nb_pca.fit(X_pca_train, y_train)
nb_pca_model

nb_pca_model.predict(X_pca_test)[0:10]
accuracy_score(y_test, y_pred)
nb_pca_score= cross_val_score(nb_pca_model, X_pca_test, y_test, cv = 10).mean()
nb_pca_score
nb = GaussianNB()
nb_model_2 = nb.fit(X_pca_train[:,:25], y_train)

nb_model_2.predict(X_pca_test[:,:25])
print(accuracy_score(y_test, y_pred))
nb_pca_rep_score=cross_val_score(nb_model_2, X_pca_test[:,:25], y_test, cv = 10).mean()
nb_pca_rep_score
print("KNN X_TRAIN SCORE : ",knn_x_train_score)
print()
print("NAIVE_BAYES X_TRAIN SCORE : ",NB_x_train_score)
print()
print("KNN PCA SCORE : ",knn_pca_score)
print()
print("NAIVE_BAYES PCA SCORE : ",nb_pca_score)
print()
print("KNN PCA REPRESENTATION SCORE : ",knn_pca_rep_score)
print()
print("NAIVE_BAYES PCA REPRESENTATION SCORE : ",nb_pca_rep_score)


import matplotlib.pyplot as plt
 
# Create bars
barWidth =0.7
bars1 = [knn_x_train_score, NB_x_train_score]
bars2 = [knn_pca_score,nb_pca_score]
bars3 = [knn_pca_rep_score,nb_pca_rep_score]

 
# The X position of bars
r1 = [1,2]
r2 = [3,4]
r3 = [5,6,]
r4 = r1 + r2 + r3
 
# Create barplot
plt.bar(r1, bars1, width = barWidth, color = (0.3,0.2,0.2,0.6), label='X_train score')
plt.bar(r2, bars2, width = barWidth, color = (0.3,0.5,0.4,0.6), label='pca-score')
plt.bar(r3, bars3, width = barWidth, color = (0.3,0.9,0.4,0.6), label='pca_rep')
# Note: the barplot could be created easily. See the barplot section for other examples.
 
# Create legend
plt.legend(borderpad=0.1)
 

plt.xticks([r+1  for r in range(len(r4))],
           ['KNN', 'NAIVE_BAYES ', '"KNN PCA', 'NAIVE_BAYES PCA', 'KNN PCA REPRESENTATION', 'NAIVE_BAYES PCA REPRESENTATION'], rotation=90)
 

plt.subplots_adjust(bottom= 0.2, top = 0.98)
 
# Show graphic
plt.show()

df=x_train
from sklearn.decomposition import PCA
pca = PCA()
X= pca.fit_transform(df)


represent=np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4)*100)
print(represent)
s=0
for i in represent:
    s+=1
    if i>=92:
        print("% 92 represent count ",i,"index :",s)
        break



X
plt.plot(np.cumsum(pca.explained_variance_ratio_[:s]))
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans
k_fit = kmeans.fit(X)
k_fit.n_clusters
k_fit.cluster_centers_
k_fit.labels_
kmeans = KMeans(n_clusters = 2)
k_fit = kmeans.fit(X)
kumeler = k_fit.labels_
plt.scatter(X[:,0], X[:,1], c = kumeler, s = 50, cmap = "viridis")

merkezler = k_fit.cluster_centers_

plt.scatter(merkezler[:,0], merkezler[:,1], c = "black", s = 200, alpha = 0.5);
from mpl_toolkits.mplot3d import Axes3D
kmeans = KMeans(n_clusters = 3)
k_fit = kmeans.fit(X)
kumeler = k_fit.labels_
print(kumeler)
merkezler = kmeans.cluster_centers_

plt.rcParams['figure.figsize'] = (16, 9)
fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(X[:, 0], X[:, 1],X[:,2],c=["green"]);


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(X[:, 0], X[:, 1], X[:, 2],c="green"),
ax.scatter(merkezler[:, 0], merkezler[:, 1], merkezler[:, 2], 
           marker='*', 
           c='red', s=5000);
# !pip install yellowbrick
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(X) 
visualizer.poof() 
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(X[:,:s]) 
visualizer.poof() 
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(df) 
visualizer.poof()
x_train
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#scale etmemiz gerekiyor 
df = StandardScaler().fit_transform(x_train)

pca_2D = PCA(n_components=2)
X_pca2D_train = pca_2D.fit_transform(df)

#bu fonksiyon modelimizdeki degiskenleri temsil etme derecesini veriyor.
pca_2D.explained_variance_ratio_
X_pca2D_train[:10]

represent=np.cumsum(np.round(pca_2D.explained_variance_ratio_, decimals = 4)*100)
print(represent)

import matplotlib.pyplot as plt

plt.scatter(X_pca2D_train[:,0],X_pca2D_train[:,1],color = ["red"],s=80)


plt.legend()
plt.xlabel("x_train_0")
plt.ylabel("x_train_1")

plt.show()

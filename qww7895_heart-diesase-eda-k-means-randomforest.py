# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import  preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score,classification_report

from sklearn.tree import export_graphviz

import six

from sklearn import tree

# Convert to png using system command (requires Graphviz)

from subprocess import call

# Display in jupyter notebook

from IPython.display import Image

from sklearn.cluster import KMeans

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

heart=pd.read_csv('../input/heart.csv')
heart.shape
# 1.3 명목형 자료의 자료형을 문자로 변환

#위의 설명을 읽고 명목형 자료를 바꿔주면서 결측치가 추가된다

heart.cp = heart.cp.replace([0,1,2,3],['no_pain','Atypical angina','Angina pain','Typical angina'])#가슴통증 유형

heart.restecg = heart.restecg.replace([0,1,2],['Left ventricular hypertrophy','nomal','ST-T wave abnormal'])#안정 심전도 결과

heart.slope = heart.slope.replace([0,1,2],['descent','plane','Increase'])

heart.thal= heart.thal.replace([0,1,2,3],[np.nan,'Resolved Defects','nomal','Resolveable Defects'])
heart.info()
heart.head(10)
heart.isnull().sum()# Null값 확인, 혈관수와 결함을 나타내는 파라메터에서 결측치를 각각 5,2개 확인
sns.pairplot(heart.dropna(), hue='target')
fig,ax=plt.subplots(1, 2, figsize = (14,7))

sns.countplot(data=heart, x='target', ax=ax[0],palette='cool')

ax[0].set_xlabel("having heart Disease?")

ax[0].set_ylabel("Count")

ax[0].set_title("Heart Disease Count")

heart['target'].value_counts().plot.pie(explode=[0,0.05],startangle=90, autopct='%0.1f%%',ax=ax[1],cmap='cool')

plt.title("Heart Disease")
print(heart['target'].value_counts())

target_df=heart[heart.target==1]

nontarget_df=heart[heart.target==0]
searchList=['sex','cp','fbs','restecg','exang','slope','ca','thal']

# 명목형 변수 리스트
def showdetail(index):

    if index is 'sex':

        plt.text(-1,-1,'0 -> female 1 -> male',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))

    elif index is 'fbs':

        plt.text(-1,-1,'0 -> Low blood sugar 1 ->  High blood sugar',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))        

    elif index is 'exang':    

        plt.text(-1,-1,'0 -> False 1 ->True',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))        

    elif index is 'ca':    

        plt.text(-1,-1,'0 ~ 3 -> blood vessel\'s num 4 -> Na',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))        



fig,ax=plt.subplots(len(searchList), 3, figsize = (20,7*len(searchList)))

axnum=1

#explode=[0.02  for x in range(len(heart[index].cat.categories))],

for index in searchList:

    plt.subplot(len(searchList),3,axnum)

    plt.title('target\'s '+index)

    target_df[index].dropna().value_counts().sort_index().plot.pie(

                                                      autopct='%0.1f%%')

    showdetail(index)

    axnum+=1

    

    plt.subplot(len(searchList),3,axnum)    

    plt.title('nontarget\'s '+index)

    nontarget_df[index].dropna().value_counts().sort_index().plot.pie(

                                                         autopct='%0.1f%%')

    showdetail(index)

    axnum+=1

    

    plt.subplot(len(searchList),3,axnum)    

    plt.title('all\'s '+index)

    heart[index].dropna().value_counts().sort_index().plot.pie(

                                                  autopct='%0.1f%%')

    showdetail(index)    

    axnum+=1
continually_index = heart.describe().columns

continually_index
continually_index = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']# 연속형 자료

fig,ax=plt.subplots(len(continually_index), 1, figsize = (14,7*len(continually_index)))

list_couint=1

for index in continually_index:

    plt.subplot(len(continually_index),1,list_couint)

    plt.title(index)

    sns.distplot(target_df[index],bins=20, kde=True, rug=True, color='red',label='target')

    sns.distplot(nontarget_df[index],bins=20,kde=True, rug=True,color='blue',label='nontarget')

    sns.distplot(heart[index],bins=20,kde=True, rug=True,color='gray',label='all')

    plt.legend()

    list_couint+=1

    

    
plt.figure(figsize=(15, 45))

plt.subplot(3,1,1)

sns.heatmap(target_df.dropna().corr(), annot = True,cmap='Blues')

plt.title('target\'s Correlation Table', fontsize = 22)

plt.subplot(3,1,2)

sns.heatmap(nontarget_df.dropna().corr(), annot = True,cmap='Blues')

plt.title('nontarget\'s Correlation Table', fontsize = 22)

plt.subplot(3,1,3)

sns.heatmap(heart.dropna().corr(), annot = True,cmap='Blues')

plt.title('All\'s Correlation Table', fontsize = 22)

target_df.dropna().corr(), nontarget_df.dropna().corr(), heart.dropna().corr()
heart1=pd.read_csv('../input/heart.csv')

heart2=pd.read_csv('../input/heart.csv')

targets=heart1.target

heart1=heart1.drop('target',axis=1)

heart1.head(10)

scaler = preprocessing.RobustScaler()

heart1=scaler.fit_transform(heart1)

fig,ax=plt.subplots(len(['age', 'trestbps', 'chol', 'thalach', 'oldpeak']), 1)

#연속형 자료들의 이상치 확인

list_couint=1

for index in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:

    plt.subplot(len(['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),1,list_couint)

    plt.title(index)

    sns.boxplot(heart[index])

    list_couint+=1
# 스케일링 -알고리즘에 적용하기에 앞서서 모델링에 알맞을 형태로 데이터를 처리

from sklearn.preprocessing import RobustScaler

scaler = preprocessing.RobustScaler() 

scaler.fit_transform(heart2)

feature = scaler.fit_transform(heart2)

def elbow(X):

    sse = []



    for i in range(1,11):

        km = KMeans(n_clusters=i,algorithm='auto', random_state=42)

        km.fit(X)

        sse.append(km.inertia_)



    plt.plot(range(1,11), sse, marker='o')

    plt.xlabel('n_clusters label') 

    plt.ylabel('SSE')

    plt.show()



elbow(feature)
import numpy as np

from sklearn.metrics import silhouette_samples

from sklearn.datasets import make_blobs

from matplotlib import cm



# 이코드에서 데이터 X와 X를 임의의 클러스터 개수로 계산한 k-means 결과인

# y

def plotSilhouette(X, y_km):

    cluster_labels = np.unique(y_km)

    n_clusters = cluster_labels.shape[0]

    silhouette_vals = silhouette_samples(X, y_km, metric = 'euclidean')

    y_ax_lower, y_ax_upper = 0, 0

    yticks = []



    for i, c in enumerate(cluster_labels):

        c_silhouette_vals = silhouette_vals[y_km == c]

        c_silhouette_vals.sort()

        y_ax_upper += len(c_silhouette_vals)

        color = cm.jet(i/n_clusters)



        plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,

                edgecolor='none', color=color)

        yticks.append((y_ax_lower + y_ax_upper)/2)

        y_ax_lower += len(c_silhouette_vals)

    

    silhoutte_avg = np.mean(silhouette_vals)

    plt.title("silhoutte_avg: {:5.2f}".format(silhoutte_avg))

    plt.axvline(silhoutte_avg, color = 'red', linestyle='--')

    plt.yticks(yticks, cluster_labels+1)

    plt.ylabel('n_clusters label')

    plt.xlabel('silhouette value')

    plt.show()



X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5,

                  shuffle=True, random_state=0)

km = KMeans(n_clusters=2, algorithm='auto', random_state=42)

km2 = KMeans(n_clusters=3, algorithm='auto', random_state=42)

km3 = KMeans(n_clusters=4, algorithm='auto', random_state=42)

y_km = km.fit_predict(heart2)

y_km2 = km2.fit_predict(heart2)

y_km3 = km3.fit_predict(heart2)

plotSilhouette(heart2, y_km)

plotSilhouette(heart2, y_km2)

plotSilhouette(heart2, y_km3)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=100,algorithm='auto').fit(heart1)

kmeans_target = len(kmeans.labels_[kmeans.labels_== 1])

kmeans_nontarget = len(kmeans.labels_[kmeans.labels_== 0])

print(kmeans_target,kmeans_nontarget)
kmeans_se=pd.Series(kmeans.labels_)

fig,ax=plt.subplots(2,2, figsize = (20,20))

plt.subplot(2,2,3)

heart['target'].dropna().value_counts().plot.pie()

plt.text(0,-1,'0 -> nontarget 1-> target',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))

plt.title('Input data')

plt.subplot(2,2,4)

kmeans_se.dropna().value_counts().plot.pie()

plt.text(0,-1,'0 -> target 1-> nontarget',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))

plt.title('K-means\'s result')





plt.subplot(2,2,1)

heart['target'].dropna().value_counts().plot.bar()

plt.text(0,1,'0 -> nontarget 1-> target',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))

plt.title('Input data')

plt.subplot(2,2,2)

kmeans_se.dropna().value_counts().plot.bar()

plt.text(0,1,'0 -> target 1-> nontarget',fontsize=15,bbox=dict(facecolor='gray', alpha=0.5))

plt.title('K-means\'s result')
# 먼저 생성된 결과를 데이터 프레임으로 변환한다.

feature_df = pd.DataFrame(heart2)
import numpy as np

from sklearn.manifold import TSNE



# 2개의 차원으로 축소

#데이터의 분류를 시각화해서 확인하고 싶으나, 

#다차원의 데이터를 눈으로 확인하기란 불가능에 가깝다. 

#단, 차원을 줄이면 가능하다. 

#차원을 축소하여 feature를 2로 줄인 후 그래프를 그려 확인해볼 수 있다.

transformed = TSNE(n_components=2).fit_transform(feature_df)

transformed.shape
xs = transformed[:,0]

ys = transformed[:,1]

colors=np.array( ['red','blue'])

bins = np.arange(xs.shape[0]) % 2

fig,ax=plt.subplots(1,1, figsize = (15,15))

plt.scatter(xs,ys,c=colors[bins])#라벨은 색상으로 분류됨

X_train, X_test, y_train, y_test = train_test_split(heart1,

                                                    targets

                                                    ,test_size=0.3

                                                    ,random_state =0

                                                   )



print(len(X_train),len(X_test),len(y_train),len(y_test))





# 학습

forest = RandomForestClassifier(n_estimators=100, random_state = 1)

forest.fit(X_train, y_train)



# 예측

y_pred = forest.predict(X_test)

#print(y_pred)

#print(list(y_test))

# from sklearn import metrics 



print('test 갯수 : ',len(y_test),'error : ',(y_test!=y_pred).sum())

print('정확도 :',accuracy_score(y_test, y_pred))

# print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))

# print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))





print(classification_report(y_test,y_pred))

#https://frhyme.github.io/machine-learning/clf_%ED%8F%89%EA%B0%80%ED%95%98%EA%B8%B0/

# precision : 예측한 값중 맞은값의 비율

# recall    : 실제 0또는 1인값이 0또는 1로 판별된 비율

# f1-score  : precision, recall 의 평가

# support   : 해당 클래스에 있는 실제 응답의 샘플 수
estimator = forest.estimators_[0]

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = heart.columns[:13],

                class_names = heart.columns[-1],

                rounded = True, proportion = False, 

                precision = 2, filled = True)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
Image(filename = 'tree.png')
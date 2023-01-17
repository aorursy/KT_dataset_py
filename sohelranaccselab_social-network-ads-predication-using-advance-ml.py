#Enivornment Setup
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
#Data Read, Data Visualization,EDA Analysis,Data Pre-Processing,Data Splitting
#Data Read

file_path = '../input/social-network-ads'

df=pd.read_csv(f'{file_path}/Social_Network_Ads.csv')
df.head()
df = df.loc[:,~df.columns.duplicated()]
import pandas_profiling
# preparing profile report



profile_report = pandas_profiling.ProfileReport(df,minimal=True)

profile_report
df.info()
X = df.drop('User ID', axis=1)
df=X.copy()
df.describe()
df.shape
df.Purchased.value_counts()
df.apply(lambda x: sum(x.isnull()),axis=0)
df.groupby("Gender").mean()
df.groupby("Age").mean()
import seaborn; seaborn.set()

df.plot();
df.corr()
def correlation_matrix(d):

    from matplotlib import pyplot as plt

    from matplotlib import cm as cm



    fig = plt.figure(figsize=(16,12))

    ax1 = fig.add_subplot(111)

    cmap = cm.get_cmap('jet', 30)

    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)

    ax1.grid(True)

    plt.title('Social_Network_Ads features correlation\n',fontsize=15)

    labels=df.columns

    ax1.set_xticklabels(labels,fontsize=9)

    ax1.set_yticklabels(labels,fontsize=9)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels

    fig.colorbar(cax, ticks=[0.1*i for i in range(-11,11)])

    plt.show()



correlation_matrix(df)
#Plotting data 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.express as px
plt.figure(figsize=(20,15))

sns.heatmap(df.corr(),annot=True,linecolor='red',linewidths=3,cmap = 'plasma')
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(df.corr(),annot=True, linewidths=.5,fmt='.1f',ax=ax)
sns.pairplot(df,diag_kind="kde")

plt.show()
i=1

plt.figure(figsize=(25,20))

for c in df.describe().columns[:]:

    plt.subplot(5,3,i)

    plt.title(f"Histogram of {c}",fontsize=10)

    plt.yticks(fontsize=12)

    plt.xticks(fontsize=12)

    plt.hist(df[c],bins=20,color='blue',edgecolor='k')

    i+=1

plt.show()
#checking the target variable countplot

sns.countplot(data=df,x = 'Purchased',palette='plasma')
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import train_test_split
le = LabelEncoder()

for col in df.select_dtypes('object').columns:

    df[col] = le.fit_transform(df[col])
df=df.copy()
df.head(n=20)
plt.figure(figsize=(20,15))

sns.heatmap(df.corr(),annot=True,linecolor='red',linewidths=3,cmap = 'plasma')
i=1

plt.figure(figsize=(35,25))

for c in df.columns[:-1]:

    plt.subplot(2,3,i)

    plt.title(f"Boxplot of {c}",fontsize=16)

    plt.yticks(fontsize=12)

    plt.xticks(fontsize=12)

    sns.boxplot(y=df[c],x=df['Purchased'])

    i+=1

plt.show()
#Numerical Columns data distribution
sns.set()

fig = plt.figure(figsize = [15,20])

cols = ['Gender', 'Age', 'EstimatedSalary', 'Purchased']

cnt = 1

for col in cols :

    plt.subplot(2,3,cnt)

    sns.distplot(df[col],hist_kws=dict(edgecolor="k", linewidth=1,color='green'),color='red')

    cnt+=1

plt.show() 
# Distplot

fig, ax2 = plt.subplots(2, 3, figsize=(16, 16))

sns.distplot(df['Gender'],ax=ax2[0][0])

sns.distplot(df['Age'],ax=ax2[0][1])

sns.distplot(df['EstimatedSalary'],ax=ax2[0][2])
sns.set()

fig = plt.figure(figsize = [15,20])

cols = ['Gender', 'Age', 'EstimatedSalary']

cnt = 1

for col in cols :

    plt.subplot(4,3,cnt)

    sns.violinplot(x="Purchased", y=col, data=df)

    cnt+=1

plt.show()
X = df.drop('Purchased', axis=1)

y = df['Purchased']
from sklearn.ensemble import ExtraTreesClassifier
ec = ExtraTreesClassifier(n_estimators=100, random_state=0)

ec.fit(X,y)
ec_series = pd.Series(ec.feature_importances_,index=X.columns)

ec_series.plot(kind = 'barh',color = 'red')
from sklearn.model_selection import  train_test_split, cross_val_score

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
#train_test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15,random_state=42)
dims = X_train.shape[1]

print(dims, 'dims')
print(y_train)
#Data Pre-Processing & Supervised machine Learning Models Performance
from sklearn.preprocessing import RobustScaler, StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#For Support vector Algorithm

from sklearn.svm import SVC

model = SVC()

model.fit(X_train,y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
#Parameter tuning

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV



grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=1)



# May take a while!

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
param_grid = {'C': [50,75,100,125,250], 'gamma': [1e-2,1e-3,1e-4,1e-5,1e-6], 'kernel': ['rbf']} 

grid = GridSearchCV(SVC(tol=1e-5),param_grid,refit=True,verbose=1)

grid.fit(X_train,y_train)
grid.best_estimator_
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))
#Using DecisionTreeClassifier



from sklearn.tree import DecisionTreeClassifier



dtree = DecisionTreeClassifier(criterion='gini',max_depth=9)
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
cm=confusion_matrix(y_test,predictions)

print(cm)
print ("Accuracy of prediction:",round((cm[0,0]+cm[1,1])/cm.sum(),3))
#Using RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
cr = classification_report(y_test,predictions)
print(cr)
#Using XGBboost Classifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
model = XGBClassifier()

model.fit(X_train, y_train)
# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]
# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
#Models performance Analysis with scaling(standard Scaler)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

list_models=[]

list_scores=[]

x_train=sc.fit_transform(X_train)

lr=LogisticRegression(max_iter=10000)

lr.fit(X_train,y_train)

pred_1=lr.predict(sc.transform(X_test))

score_1=accuracy_score(y_test,pred_1)

list_scores.append(score_1)

list_models.append('LogisticRegression')
score_1
from sklearn.neighbors import KNeighborsClassifier

list_1=[]

for i in range(1,11):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    preds=knn.predict(sc.transform(X_test))

    scores=accuracy_score(y_test,preds)

    list_1.append(scores)
sns.lineplot(x=list(range(1,11)),y=list_1)
list_scores.append(max(list_1))

list_models.append('KNeighbors Classifier')
print(max(list_1))
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()

rfc.fit(x_train,y_train)

pred_2=rfc.predict(sc.transform(X_test))

score_2=accuracy_score(y_test,pred_2)

list_models.append('Randomforest Classifier')

list_scores.append(score_2)
score_2
from sklearn.svm import SVC

svm=SVC()

svm.fit(x_train,y_train)

pred_3=svm.predict(sc.transform(X_test))

score_3=accuracy_score(y_test,pred_3)

list_scores.append(score_3)

list_models.append('Support vector machines')
score_3
from xgboost import XGBClassifier

xgb=XGBClassifier()

xgb.fit(x_train,y_train)

pred_4=xgb.predict(sc.transform(X_test))

score_4=accuracy_score(y_test,pred_4)

list_models.append('XGboost')

list_scores.append(score_4)
score_4
plt.figure(figsize=(12,5))

plt.bar(list_models,list_scores)

plt.xlabel('classifiers')

plt.ylabel('accuracy scores')

plt.show()
#Esamble models: # importing libraries 
from sklearn.decomposition import PCA

from sklearn.utils import resample

import tensorflow as t



from sklearn.ensemble import VotingClassifier ,BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression,RidgeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,RobustScaler,StandardScaler

import xgboost 

from sklearn.svm import LinearSVR,SVR



from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold,train_test_split

from sklearn.metrics import accuracy_score 

from numpy import mean,std

from sklearn.feature_selection import SelectKBest,f_regression

from sklearn.linear_model import LinearRegression,BayesianRidge,ElasticNet,Lasso,SGDRegressor,Ridge

from sklearn.kernel_ridge import KernelRidge



from sklearn.pipeline import make_pipeline,Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.decomposition import KernelPCA

from sklearn.ensemble import ExtraTreesRegressor,GradientBoostingRegressor,RandomForestRegressor,VotingClassifier

from sklearn.model_selection import cross_val_score,KFold,GridSearchCV,RandomizedSearchCV,StratifiedKFold,train_test_split

from sklearn.base import BaseEstimator,clone,TransformerMixin,RegressorMixin



from scipy.stats import skew

from scipy.stats.stats import pearsonr

from matplotlib import pyplot



from matplotlib.pyplot import figure

figure(num=2, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')

%matplotlib inline

seed = 2020

np.random.seed(seed)
# Ensemble of Models 

estimator = [] 

estimator.append(('LR',LogisticRegression(solver ='lbfgs',multi_class ='multinomial',max_iter = 200))) 

estimator.append(('SVC', SVC(gamma ='auto', probability = True))) 

estimator.append(('DTC', DecisionTreeClassifier()))



estimator.append(('RandomForestClassifier', RandomForestClassifier()))  

estimator.append(('AdaBoostClassifier', AdaBoostClassifier())) 

estimator.append(('GradientBoostingClassifier', GradientBoostingClassifier())) 

 

estimator.append(('XGBClassifier', XGBClassifier())) 

estimator.append(('BaggingClassifier', BaggingClassifier())) 

estimator.append(('ExtraTreesClassifier', ExtraTreesClassifier()))  



estimator.append(('GaussianNB', GaussianNB()))

estimator.append(('KNeighborsClassifier', KNeighborsClassifier()))

# Voting Classifier with hard voting 

hard_voting = VotingClassifier(estimators = estimator, voting ='hard') 

hard_voting.fit(X_train, y_train) 

y_pred = hard_voting.predict(X_test)  
# accuracy_score metric to predict Accuracy 

score = accuracy_score(y_test, y_pred) 

print("Hard Voting Score % d" % score)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
# Voting Classifier with soft voting 

soft_voting = VotingClassifier(estimators = estimator, voting ='soft') 

soft_voting.fit(X_train, y_train) 

y_pred = soft_voting.predict(X_test) 
# Using accuracy_score 

score = accuracy_score(y_test, y_pred) 

print("Soft Voting Score % d" % score) 
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
ada_boost = AdaBoostClassifier(random_state=2020)

ada_boost.fit(X_train, y_train)

ada_boost.score(X_test,y_test)
grad_boost= GradientBoostingClassifier(learning_rate=0.03,random_state=2020)

grad_boost.fit(X_train, y_train)

grad_boost.score(X_test,y_test)
xgb_boost=XGBClassifier(random_state=2020,learning_rate=0.005)

xgb_boost.fit(X_train, y_train)

xgb_boost.score(X_test,y_test)
# Create classifiers

rf = RandomForestClassifier()

et = ExtraTreesClassifier()

knn = KNeighborsClassifier()

svc = SVC()

rg = RidgeClassifier()

clf_array = [rf, et, knn, svc, rg]

for clf in clf_array:

    vanilla_scores = cross_val_score(clf, X, y, cv=5, n_jobs=-1)

    bagging_clf = BaggingClassifier(clf,max_samples=0.4, max_features=10, random_state=seed)

    bagging_scores = cross_val_score(bagging_clf, X, y, cv=5,n_jobs=-1)

    

    print ("Mean of: {1:.3f}, std: (+/-) {2:.3f} [{0}]".format(clf.__class__.__name__,vanilla_scores.mean(), vanilla_scores.std()))

    print ("Mean of: {1:.3f}, std: (+/-) {2:.3f} [Bagging {0}]\n".format(clf.__class__.__name__,bagging_scores.mean(), bagging_scores.std()))
from sklearn.ensemble import VotingClassifier

clf = [rf, et, knn, svc, rg]

eclf = VotingClassifier(estimators=[('Random Forests', rf), ('Extra Trees', et), ('KNeighbors', knn), ('SVC', svc), ('Ridge Classifier', rg)], voting='hard')

for clf, label in zip([rf, et, knn, svc, rg, eclf], ['Random Forest', 'Extra Trees', 'KNeighbors', 'SVC', 'Ridge Classifier', 'Ensemble']):

    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#Un-Supervised machine Learning Models Performance
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



from sklearn.preprocessing import StandardScaler



np.random.seed(5)
"""What is clustering?¶

Clustering is an unsupervized learning technique where you take the entire dataset and find the "groups of similar entities" within the dataset. Hence there is no labels within the dataset.



Useful for organizing very large dataset into meaningful clusters that can be useful and actions can be taken upon. For example, take entire customer base of more than 1M records and try to group into high value customers, low value customers and so on.



What questions does clustering typically tend to answer?



Types of pages are there on the Web?

Types of customers are there in my market?

Types of people are there on a Social network?

Types of E-mails in my Inbox?

Types of Genes the human genome has?

From clustering to classification

Clustering is base of all the classification problems. Initially, say we have a large ungrouped number of users in a new social media platform. We know for certain that the number of users will not be equal to the number of groups in the social media, and it will be reasonably finite.

Even though each user can vary in fine-grain, they can be reasonably grouped into clusters.

Each of these grouped clusters become classes when we know what group each of these users fall into.



"""
#Partition clustering
standard_scalar = StandardScaler()

data_scaled = standard_scalar.fit_transform(df)

df = pd.DataFrame(data_scaled, columns=df.columns)

df.head()
from sklearn.cluster import KMeans



km = KMeans(init="random", n_clusters=2)

km.fit(df)
km.labels_
km.cluster_centers_
# k-means determine k

distortions = []

K = range(1, 20)

for k in K:

    kmeanModel = KMeans(n_clusters=k)

    kmeanModel.fit(df)

    distortions.append(kmeanModel.inertia_)

    

# Plot the elbow

plt.plot(K, distortions, 'bx-')

plt.xlabel('No of clusters (k)')

plt.ylabel('Distortion')

plt.title('The Elbow Method showing the optimal k')

plt.show()
estimators = [('k_means_5', KMeans(n_clusters=5, init='k-means++')),

              ('k_means_2', KMeans(n_clusters=2, init='k-means++')),

              ('k_means_bad_init', KMeans(n_clusters=2, n_init=1, init='random'))]



fignum = 1

titles = ['5 clusters', '2 clusters', '2 clusters, bad initialization']



for name, est in estimators:

    fig = plt.figure(fignum, figsize=(8, 6))

    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    est.fit(df)

    labels = est.labels_



    ax.scatter(df.values[:, 3], df.values[:, 0], df.values[:, 2], c=labels.astype(np.float), edgecolor='k')



    ax.w_xaxis.set_ticklabels([])

    ax.w_yaxis.set_ticklabels([])

    ax.w_zaxis.set_ticklabels([])

    ax.set_xlabel('Gender')

    ax.set_ylabel('Age')

    ax.set_zlabel('EstimatedSalary')

    ax.set_title(titles[fignum - 1])

    ax.dist = 12

    fignum = fignum + 1

#Hierarchical Clustering or Agglomerative clustering.
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering().fit(df)

clustering
clustering.labels_
from scipy.cluster.hierarchy import dendrogram



def plot_dendrogram(model, **kwargs):

    # Create linkage matrix and then plot the dendrogram



    # create the counts of samples under each node

    counts = np.zeros(model.children_.shape[0])

    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):

        current_count = 0

        for child_idx in merge:

            if child_idx < n_samples:

                current_count += 1  # leaf node

            else:

                current_count += counts[child_idx - n_samples]

        counts[i] = current_count



    linkage_matrix = np.column_stack([model.children_, model.distances_,

                                      counts]).astype(float)



    # Plot the corresponding dendrogram

    dendrogram(linkage_matrix, **kwargs)

    

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)



model = model.fit(df)



plt.figure(fignum, figsize=(10, 6))

plt.title('Hierarchical Clustering Dendrogram')

# plot the top three levels of the dendrogram

plot_dendrogram(model, truncate_mode='level', p=3)

plt.xlabel("Number of points in node (or index of point if no parenthesis).")

plt.show()

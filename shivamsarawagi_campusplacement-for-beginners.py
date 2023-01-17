import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report,roc_auc_score,roc_curve

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from mpl_toolkits.mplot3d import Axes3D 

from sklearn.linear_model import LogisticRegression

import scipy

from scipy.spatial.distance import pdist,cdist

from scipy.cluster.hierarchy import dendrogram,linkage

from scipy.cluster.hierarchy import fcluster

from scipy.cluster.hierarchy import cophenet

from sklearn.cluster import AgglomerativeClustering

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn import tree

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
df.head()
df.info()
df.isnull().sum()
#As salary is not relavent

df.drop(columns=["salary","sl_no"],inplace=True)
#dist plot

fig,ax = plt.subplots(2,3, figsize=(10,10))               # 'ax' has references to all the four axes

sns.distplot(df['ssc_p'], ax = ax[0,0]) 

sns.distplot(df['hsc_p'], ax = ax[0,1]) 

sns.distplot(df['degree_p'], ax = ax[0,2]) 

sns.distplot(df['etest_p'], ax = ax[1,0]) 

sns.distplot(df['mba_p'], ax = ax[1,1]) 

plt.show()
#count plot

total_records= len(df)

columns = ["gender","ssc_b","hsc_b","hsc_s","degree_t",

           "workex","specialisation"]

plt.figure(figsize=(12,8))

j=0

for i in columns:

    j +=1

    plt.subplot(4,2,j)

    ax1 = sns.countplot(data=df,x= i,hue="status")

    if(j==8 or j== 7):

        plt.xticks( rotation=90)

    for p in ax1.patches:

        height = p.get_height()

        ax1.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}'.format(height/total_records,0),

                ha="center",rotation=0) 

plt.subplots_adjust(bottom=-0.9, top=2)

plt.show()
#Feature Encoding

df["gender"] = df.gender.map({"M":0,"F":1})

df["hsc_s"] = df.hsc_s.map({"Commerce":0,"Science":1,"Arts":2})

df["degree_t"] = df.degree_t.map({"Comm&Mgmt":0,"Sci&Tech":1, "Others":2})

df["workex"] = df.workex.map({"No":0, "Yes":1})

df["status"] = df.status.map({"Not Placed":0, "Placed":1})

df["specialisation"] = df.specialisation.map({"Mkt&HR":0, "Mkt&Fin":1})

df["ssc_b"] = df.ssc_b.map({"Others":0,"Central":1})

df["hsc_b"] = df.hsc_b.map({"Others":0,"Central":1})
#heat map

plt.figure(figsize=(12,8))

sns.heatmap(data=df.corr(),cmap="YlGnBu")
df.info()
df.head()
Y = df["status"]

X = df.drop(columns=["status"])

#split data

x_train,x_test,y_train,y_test = train_test_split(X, 

                                                         Y, 

                                                         train_size= 0.80,

                                                         random_state=0);
Model = []

Accuracy = []
LR = LogisticRegression(multi_class='auto',max_iter=1000)

LR.fit(x_train,y_train)

y_pred = LR.predict(x_test)
Model.append('Logistic Regression')

Accuracy.append(accuracy_score(y_test,y_pred))
params = {

    

    'n_neighbors': range(1,25),

    'weights': ['uniform','distance'],

    'algorithm': ['ball_tree','kd_tree','brute','auto'],

    'p': [1,2,3]

}



knn = KNeighborsClassifier()



gs = GridSearchCV(estimator=knn,n_jobs=-1,cv=5,param_grid=params)

gs.fit(X,Y)
gs.best_params_
knn = KNeighborsClassifier(**gs.best_params_)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)

Model.append('KNN')

Accuracy.append(accuracy_score(y_test,y_pred))
nb = GaussianNB()

nb.fit(x_train,y_train)

y_pred = nb.predict(x_test)

Model.append('Naive')

Accuracy.append(accuracy_score(y_test,y_pred))
params = {

    

    'criterion':['gini','entropy'],

    'splitter':['best','random'],

    'max_depth':range(1,10,1),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



dt = DecisionTreeClassifier()



gs = GridSearchCV(estimator=dt,n_jobs=-1,cv=3,param_grid=params)

gs.fit(X,Y)
gs.best_params_
dt = DecisionTreeClassifier(**gs.best_params_)

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

Model.append('Decision Tree')

Accuracy.append(accuracy_score(y_test,y_pred))
params = {

    

    'n_estimators':range(10,100,10),

    'criterion':['gini','entropy'],

    'max_depth':range(2,10,1),

    'max_leaf_nodes':range(2,10,1),

    'max_features':['auto','log2']

    

}



rf = RandomForestClassifier()



gs = GridSearchCV(estimator=rf,param_grid=params,cv=3)

gs.fit(X,Y)
gs.best_params_
rf = RandomForestClassifier(**gs.best_params_)

rf.fit(x_train,y_train)

y_pred = rf.predict(x_test)

Model.append('Random Forrest')

Accuracy.append(accuracy_score(y_test,y_pred))
gb_Boost = GradientBoostingClassifier(n_estimators=100,learning_rate=0.01)

gb_Boost.fit(x_train,y_train)

y_pred = rf.predict(x_test)

Model.append('Gradient Boosting')

Accuracy.append(accuracy_score(y_test,y_pred))
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(x_train)

X_test_scaled = scaler.fit_transform(x_test)
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test_scaled)

Model.append('SVC')

Accuracy.append(accuracy_score(y_test,y_pred))
result = pd.DataFrame({'Model':Model,'Accuracy':Accuracy})

result
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")
df.head()
df.describe()
df.info()
df.rename(columns = {"Annual Income (k$)": "Income", 

                     "Spending Score (1-100)":"SpendingScore"}, 

                                 inplace = True) 
df.head()
f,ax1 = plt.subplots(figsize =(30,10))

sns.pointplot(x='CustomerID',y='Income',data=df,color='lime',alpha=0.8)
g= sns.jointplot("SpendingScore","Income",data=df,size=5,ratio=3,color="r")
sns.catplot(x="Gender",y="Income",data=df, kind = "bar", height = 6)

plt.show()
sns.catplot(x="Gender",y="SpendingScore",data=df, kind = "bar", height = 6)

plt.show()
plt.hist(df["Age"],bins=10)

plt.title("Age by Histogram")

plt.show()
g= sns.jointplot("Age","Income",data=df,height=5,ratio=3,color="r")
g = sns.jointplot(df.Age, df.SpendingScore, kind="kde",height=7)

plt.show()
sns.pairplot(df.iloc[:,1:])

plt.show()
sns.heatmap(df.iloc[:,1:].corr(),annot=True,fmt = ".2f")

plt.show
df_unsupervised = df.iloc[:,3:]
from sklearn.cluster import KMeans

wcss=[]

for k in range(1,15):

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(df_unsupervised)

    wcss.append(kmeans.inertia_)

    

plt.plot(range(1,15),wcss)

plt.show()
kmeans2 = KMeans(n_clusters=5)



clusters = kmeans2.fit_predict(df_unsupervised)
plt.scatter(df_unsupervised.Income[clusters == 0],df_unsupervised.SpendingScore[clusters == 0], c = 'red', label = 'Cluster 1')

plt.scatter(df_unsupervised.Income[clusters == 1], df_unsupervised.SpendingScore[clusters == 1], c = 'blue', label = 'Cluster 2')

plt.scatter(df_unsupervised.Income[clusters == 2], df_unsupervised.SpendingScore[clusters == 2],c = 'green', label = 'Cluster 3')

plt.scatter(df_unsupervised.Income[clusters == 3], df_unsupervised.SpendingScore[clusters == 3], c = 'cyan', label = 'Cluster 4')

plt.scatter(df_unsupervised.Income[clusters == 4], df_unsupervised.SpendingScore[clusters == 4], c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1],  c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()
from scipy.cluster.hierarchy import linkage, dendrogram



merg = linkage(df_unsupervised.iloc[:100,:],method="ward")



dendrogram(merg,leaf_rotation = 90)



plt.xlabel("data points")



plt.ylabel("ceuclidean distance")



plt.show()


from sklearn.cluster import AgglomerativeClustering



hiyertical_cluster = AgglomerativeClustering(n_clusters =3,affinity="euclidean",linkage="ward")



cluster = hiyertical_cluster.fit_predict(df_unsupervised)



df_unsupervised["label"]=cluster





plt.scatter(df_unsupervised.Income[df_unsupervised["label"] == 0],df_unsupervised.SpendingScore[df_unsupervised["label"] == 0], c = 'red', label = 'Cluster 1')

plt.scatter(df_unsupervised.Income[df_unsupervised["label"] == 1], df_unsupervised.SpendingScore[df_unsupervised["label"] == 1], c = 'blue', label = 'Cluster 2')

plt.scatter(df_unsupervised.Income[df_unsupervised["label"] == 2], df_unsupervised.SpendingScore[df_unsupervised["label"] == 2],c = 'green', label = 'Cluster 3')

plt.scatter(kmeans2.cluster_centers_[:, 0], kmeans2.cluster_centers_[:, 1],  c = 'yellow', label = 'Centroids')

plt.title('Clusters of customers')

plt.xlabel('Annual Income (k$)')

plt.ylabel('Spending Score (1-100)')

plt.legend()

plt.show()

plt.show()
df_supervised= df.drop(labels=["CustomerID"],axis=1)

df_supervised["Gender"] = [1 if i=="Male" else 0 for i in df_supervised["Gender"]]
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
df_supervised.head()
x = df_supervised.iloc[:,1:].values

y = df_supervised.iloc[:,0].values

print(x.shape)

print(y.shape)
x = (x-np.min(x))/(np.max(x)-np.min(x))
x_train , x_test, y_train , y_test = train_test_split(x,y,test_size=0.1,random_state=42)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
random_state=42

classifier = [DecisionTreeClassifier(random_state = random_state),

             SVC(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             KNeighborsClassifier(),

             GaussianNB()]

             

dt_param_grid={"min_samples_split":range(10,500,20),

              "max_depth":range(1,20,2)}



svc_param_grid ={"kernel":["rbf"],

               "gamma":[0.001,0.01,0.1,1],

               "C":[1,10,50,100,200,300,1000]}



rf_param_grid={"max_features":[1,3,10],

              "min_samples_split":[2,3,10],

              "min_samples_leaf":[1,3,10],

              "bootstrap":[False],

              "n_estimators":[100,300],

              "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}

gauss_grid = {}



classifier_param = [dt_param_grid,

                   svc_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid,

                    gauss_grid

                   ]
cv_results = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(x_train,y_train)

    cv_results.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_results[i])
nb = GaussianNB()

from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(estimator=nb,X = x_train,y = y_train,cv=10)



print("mean =", np.mean(accuracies))
cv_result = pd.DataFrame({"Cross Validation Means":cv_results, "ML Models":["DecisionTreeClassifier", "SVM","RandomForestClassifier",

             "LogisticRegression",

             "KNeighborsClassifier",

             "Gaussian"]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_result)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
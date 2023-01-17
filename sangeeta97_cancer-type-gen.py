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
train_df= pd.read_csv('/kaggle/input/gene-expression/data_set_ALL_AML_train.csv')
test_df= pd.read_csv('/kaggle/input/gene-expression/data_set_ALL_AML_independent.csv')
# removing all call columns from data frame

train_columns = [col for col in train_df if "call" not in col]



train_df = train_df[train_columns]

train_df = train_df.set_index("Gene Accession Number").T
import re
train_d= [col for col in train_df if not re.match("^AFFX", col)]
train_df = train_df[train_d]
train_df = train_df.drop(["Gene Description"])
train_df.head()
def transformation(df):

    df_columns = [col for col in df.columns if "call" not in col]

    df = df[df_columns]

    df = df.set_index("Gene Accession Number").T

    dftag= [col for col in df if not re.match("^AFFX", col)]

    df = df[dftag]

    df = df.drop(["Gene Description"])

    return df

    

    
test_df= transformation(test_df)
train_df= train_df.replace(np.inf, np.nan)
train_df.isnull().sum()[train_df.isnull().sum()> 0]
cancer_types = pd.read_csv('../input/gene-expression/actual.csv')
# Reset the index. The indexes of two dataframes need to be the same before you combine them

train_df = train_df.reset_index(drop=True)



# Subset the first 38 patient's cancer types

ct_train = cancer_types[cancer_types.patient <= 38].reset_index(drop=True)



# Combine dataframes for first 38 patients: Patient number + cancer type + gene expression values

train_df = pd.concat([ct_train,train_df], axis=1)









train_df.tail()
def label(df):

    df= df.reset_index(drop= True)

    ct_test = cancer_types[cancer_types.patient > 38].reset_index(drop=True)

    df = pd.concat([ct_train,df], axis=1)

    return df

    
train_df.isnull().sum()
train_df.dtypes
test_df= label(test_df)
train_df['cancer']= train_df.cancer.map({'ALL': 0, 'AML': 1})
test_df['cancer']= test_df.cancer.map({'ALL': 0, 'AML': 1})
for col in train_df.columns:

    train_df[col]= pd.to_numeric(train_df[col])

    
for col in test_df.columns:

    test_df[col]= pd.to_numeric(test_df[col])

    
train_df['cancer']= train_df['cancer'].astype('category')
statistic= train_df.groupby('cancer').describe()
zero= train_df[train_df.cancer== 0].describe().T
zero.columns
zero['avg']= zero['std']/zero['mean']
one = train_df[train_df.cancer== 1].describe().T
one['avg']= one['std']/one['mean']
kl= (zero['avg']-one['avg']).abs().sort_values(ascending= False)[5:15].index
outcomes = train_df.groupby('cancer').size()

outcomes.plot(kind = 'bar')
x_train = train_df.iloc[:,2:]

y_train = train_df.iloc[:,1]
x_test = test_df.iloc[:,2:]

y_test = test_df.iloc[:,1]
p1= x_train[kl]
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

print(plt.style.available)
plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(p1.corr(), cmap = 'Wistia', annot = True)

plt.title('Heatmap for the Data', fontsize = 20)

plt.savefig('correlation');
sns.pairplot(p1)

plt.title('Pairplot for the Data', fontsize = 20)

plt.savefig('correlation1');
#Group-wise Plotting
p2= pd.concat([train_df[['cancer']], train_df[kl]], axis= 1)



p3= p2[p2.cancer== 0]

p4= p2[p2.cancer== 1]

p3= p3.drop('cancer', axis= 1)

p4= p4.drop('cancer', axis= 1)

k= pd.DataFrame(p3.stack())

k= k.reset_index()



k.columns= ['patient', 'fields', 'values']

p4= pd.DataFrame(p4.stack())

p4= p4.reset_index()

p4.columns= ['patient', 'fields', 'values']





sns.factorplot('fields','values',data=p4, size=6, aspect=1.7, legend=False)

                                                                                                                                                                                                                                                                                                                                                                

sns.factorplot('fields','values',data=k, size=6, aspect=1.7, legend=False)
sns.boxplot(x="fields", y="values", data=p4)

plt.title('boxplot of control')

plt.show()

plt4= sns.boxplot(x="fields", y="values", data=k)
#from sklearn import preprocessing

#scaled = pd.DataFrame(preprocessing.scale(x_train))



import matplotlib.pyplot as plt
# Using the elbow method to find  the optimal number of clusters

from sklearn.cluster import KMeans

wcss = []

for i in range(1,11):

    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)

    kmeans.fit(x_train)

    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()



# Applying k-means to the cars dataset

kmeans = KMeans(n_clusters=2,init='k-means++',max_iter=300,n_init=10,random_state=0) 

y_kmeans = kmeans.fit_predict(x_train)



X = (x_train).as_matrix(columns=None)

y_kmeans== 0, 1
y_kmeans
# Visualising the clusters

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1],s=100,c='red',label='AML')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1],s=100,c='blue',label='ALL')



plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')

plt.title('Clusters of two cancer type (ALL AND AML)')

plt.legend()

plt.show()
from sklearn.neighbors import KNeighborsClassifier

def knn_pred(train_predictors, train_outcome, k_range, test_predictors):

    #train_predictors and train_outcome should both be from training split while test_predictors should be from test split

    y_pred = []

    for i in k_range:

        knn = KNeighborsClassifier(n_neighbors = i)

        knn.fit(train_predictors, train_outcome)

        y_pred.append(knn.predict(test_predictors))

    return y_pred
from sklearn.metrics import accuracy_score
#function compares KNN accuracy at different levels of K

def knn_accuracy(pred, k_range, test_outcome):

    #pred represents predicted values while test_outcome represents the values from the test set

    accuracy_chart = []

    for i in range(len(k_range)):

        accuracy_chart.append((accuracy_score(test_outcome, pred[i])))

    return accuracy_chart
x_test= x_test.fillna(method= 'ffill')
train_range = range(2, 20, 2)

sample_pred = knn_pred(x_train, y_train, train_range, x_test)

accuracy = knn_accuracy(sample_pred, train_range, y_test)

plt.figure(figsize=(10, 8))

plt.bar(train_range, accuracy)

plt.ylim(0,1)

plt.xlim(0,20)

plt.locator_params(axis='y', nbins=20)

plt.locator_params(axis = 'x', nbins = 10)

plt.ylabel("Accuracy")

plt.xlabel("Number of Neighborhoods")
x_train.shape
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(x_train,method="complete", metric= 'cosine')

dendrogram(merg,leaf_rotation = 90)

plt.xlabel("data points")

plt.ylabel("euclidean distance")

plt.show()
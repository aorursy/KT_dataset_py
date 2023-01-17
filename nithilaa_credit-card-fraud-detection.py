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
dataset = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
dataset.head()
dataset.shape
dataset.info()
dataset.columns
dataset.describe()
dataset.isnull().sum()
import seaborn as sns 

import matplotlib.pyplot as plt
y = dataset['Class']

y.value_counts()
X = dataset.iloc[:,:-1]

X.shape
count_classes = pd.value_counts(dataset['Class'], sort=True).sort_index()

count_classes.plot(kind='bar')

plt.title("Fraud Class Histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")

plt.show()
corr_mtx = dataset.corr()

f, ax = plt.subplots(figsize=(16, 14))

ax = sns.heatmap(corr_mtx,annot=False,cmap="YlGnBu")
print(corr_mtx['Class'].sort_values(ascending = False)) 
X.hist(figsize=(20,21))

plt.show()
dataset_new = dataset.drop(columns=['Time', 'V1', 'V2', 'V3','V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18','V20','V22', 'V23','V24', 'V25', 'V26', 'V27', 'V28', 'Amount'],axis=1)
dataset_new.sample(10)
dataset_new["Class"].value_counts()
from sklearn.model_selection import train_test_split 



y = dataset_new.iloc[:,-1]

X = dataset_new.iloc[:,:-1]



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.cluster import KMeans



ks = range(1, 6)

inertias = []

for k in ks:

    model = KMeans(n_clusters=k)

    model.fit(X_train)

    inertias.append(model.inertia_)

    

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()
from sklearn.neighbors import KNeighborsClassifier



model = KNeighborsClassifier(n_neighbors=4)

k_labels = model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)
from sklearn.model_selection import GridSearchCV

#create new a knn model

knn2 = KNeighborsClassifier()

#create a dictionary of all values we want to test for n_neighbors

param_grid = {"n_neighbors": np.arange(1, 25)}

#use gridsearch to test all values for n_neighbors

knn_gscv = GridSearchCV(knn2, param_grid, cv=5)

#fit model to data

knn_gscv.fit(X, y)
knn_gscv.predict(X_test)
knn_gscv.best_score_
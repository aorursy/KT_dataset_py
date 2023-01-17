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
data = pd.read_csv("/kaggle/input/hospital-bed-capacity-and-covid19/HRR Scorecard_ 20 _ 40 _ 60 - 40 Population.csv").copy()
hrr=data.drop(data.columns[9:],axis=1)
hrr.head()
hrr.isnull().sum()
hrr=hrr.dropna()
hrr.isnull().sum()
hrr.index = hrr.iloc[:,0]
hrr.head()
hrr = hrr.iloc[:,1:]
hrr.tail()
hrr.index.name=None
hrr.head()
hrr.info()
hrr = hrr.replace(',','', regex=True)

cols = hrr.columns.drop('Available ICU Beds')

hrr[cols] = hrr[cols].apply(pd.to_numeric, errors='coerce')
hrr.dtypes
hrr.hist(figsize = (10,10));
df=hrr.copy()
df['THB_per_adult'] = df['Total Hospital Beds'] / df['Adult Population']
df['AHB_per_adult'] = df['Available Hospital Beds'] / df['Adult Population']
df['PAHB_per_adult'] = df['Potentially Available Hospital Beds*'] / df['Adult Population']
df['THB_per_65+'] = df['Total Hospital Beds'] / df['Population 65+']
df['AHB_per_65+'] = df['Available Hospital Beds'] / df['Population 65+']
df['PAHB_per_65+'] = df['Potentially Available Hospital Beds*'] / df['Population 65+']
df['TIB_per_adult'] = df['Total ICU Beds'] / df['Adult Population']
df['AIB_per_adult'] = df['Available ICU Beds'] / df['Adult Population']
df['PAIB_per_adult'] = df['Potentially Available ICU Beds*'] / df['Adult Population']
df['TIB_per_65+'] = df['Total ICU Beds'] / df['Population 65+']
df['AIB_per_65+'] = df['Available ICU Beds'] / df['Population 65+']
df['PAIB_per_65+'] = df['Potentially Available ICU Beds*'] / df['Population 65+']
df= df.drop(df.columns[:8],axis=1)
df.head(10)
df.hist(figsize = (10,10));
df.describe().T
from sklearn.preprocessing import StandardScaler

scaled_df = StandardScaler().fit_transform(df)
scaled_df= pd.DataFrame(scaled_df)
scaled_df.head()
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca_fit = pca.fit_transform(scaled_df)
reduced_df = pd.DataFrame(data = pca_fit, 
                          columns = ["variable_1","variable_2"], index=df.index)
reduced_df.head()
pca.explained_variance_ratio_
from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(2,20))
visualizer.fit(reduced_df) 
visualizer.poof();
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
k_fit = kmeans.fit(reduced_df)
clusters = k_fit.labels_
clusters
import matplotlib.pyplot as plt
import seaborn as sns
plt.scatter(reduced_df.iloc[:,0], reduced_df.iloc[:,1], c = clusters, s = 50, cmap = "viridis")

centers = k_fit.cluster_centers_

plt.scatter(centers[:,0], centers[:,1], c = "black", s = 200, alpha = 0.5);
df["class"] = clusters
df["class"].value_counts().plot.barh();
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
data = df.copy()
y = data["class"]
X = data.drop(['class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.30, 
                                                    random_state=42)
knn_params = {"n_neighbors": np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, knn_params, cv=10)
knn_cv.fit(X_train, y_train)
print("best parameter: " + str(knn_cv.best_params_))
knn = KNeighborsClassifier(3)
knn_tuned = knn.fit(X_train, y_train)
y_pred = knn_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
cf_matrix= confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix, annot=True,fmt='.3g');
catb_params = {
    'iterations': [200,500],
    'learning_rate': [0.01,0.05, 0.1],
    'depth': [3,5,8] }
catb = CatBoostClassifier()
catb_cv_model = GridSearchCV(catb, catb_params, cv=5, n_jobs = -1, verbose = 2)
catb_cv_model.fit(X_train, y_train)
catb_cv_model.best_params_
catb = CatBoostClassifier(iterations = 500, 
                          learning_rate = 0.05, 
                          depth = 8)

catb_tuned = catb.fit(X_train, y_train)
y_pred = catb_tuned.predict(X_test)
y_pred = catb_tuned.predict(X_test)
accuracy_score(y_test, y_pred)
cf_matrix= confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix, annot=True,fmt='.3g');
nb = GaussianNB()
nb_model = nb.fit(X_train, y_train)
nb_model
y_pred = nb_model.predict(X_test)
accuracy_score(y_test, y_pred)
cross_val_score(nb_model, X_test, y_test, cv = 10).mean()
cf_matrix= confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix, annot=True,fmt='.3g');
models = [
    knn_tuned,
    catb_tuned,
    nb_model,   
]

for model in models:
    names = model.__class__.__name__
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(names + ":" )
    print("Accuracy: {:.4%}".format(accuracy))
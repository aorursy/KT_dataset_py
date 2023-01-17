import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestRegressor
heart = pd.read_csv("../input/heart.csv")

heart.head()
print(heart.dtypes)
sns.pairplot(heart[['age','sex','cp','chol','slope']], hue='slope', palette='afmhot',size=1.4)
heart.target.value_counts()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(

        heart.drop('target', 1), 

        heart['target'], 

        test_size = 0.3, 

        random_state=10

        ) 
X_train.shape                        

                      

                     

                      
X_test.shape 
y_test.shape  
y_train.shape 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=1000)

clf.fit(X_train, y_train)

from sklearn.metrics import log_loss
y_test_pred = clf.predict_proba(X_test)

log_loss(y_test, y_test_pred)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error
rf = RandomForestRegressor(n_jobs=-1, n_estimators=50, verbose=3)

rf.fit(X_train, y_train)

heart_preds = rf.predict(X_test)

print(mean_absolute_error(y_test, heart_preds))
clf.feature_importances_ 

clf.feature_importances_.size
(heart_preds == y_test).sum()/y_test.size  # Check the accuracy
from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
############################ Feature creation using kmeans ####################

# Create a StandardScaler instance

se = StandardScaler()
#fit() and transform() in one step

heart = se.fit_transform(heart)
heart.shape
#  Perform kmeans using 13 features.

#     No of centroids is no of classes in the 'target'

centers = y_train.nunique()  

centers       # 2
from sklearn.cluster import KMeans  
# Begin clustering

#First create object to perform clustering

kmeans = KMeans(n_clusters=centers, # How many

                n_jobs = 5)         # Parallel jobs for n_init
kmeans.fit(heart[:, : 13])

kmeans.labels_

kmeans.labels_.size
from sklearn.preprocessing import OneHotEncoder

# Create an instance of OneHotEncoder class

ohe = OneHotEncoder(sparse = False)
ohe.fit(kmeans.labels_.reshape(-1,1)) 
# Transform data now

dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))

dummy_clusterlabels

dummy_clusterlabels.shape 
#  We will use the following as names of new two columns

#      We need them at the end of this code



k_means_names = ["k" + str(i) for i in range(2)]

k_means_names
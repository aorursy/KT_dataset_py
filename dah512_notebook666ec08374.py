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
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor

from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

from sklearn.metrics import mean_squared_error as MSE

from sklearn.tree import DecisionTreeRegressor



#Import Normalizer() and make_pipeline()

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



#Generate a synthetic dataset



rand = np.random.random((3000, 380))

print(rand)



rand.shape

metadata = rand

print("METADATA SHAPE", metadata.shape)



# Convert numpy array, rand into a Pandas dataframe



metadata_df = pd.DataFrame(metadata)

X= metadata_df

print("X SHAPE", X.shape)

y = 1/X

print("Y SHAPE", y.shape)







# Import necessary modules

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)



# Create the regressor: reg_all

reg = LinearRegression()



# Fit the regressor to the training data

reg_model= reg.fit(X_train, y_train)



# Predict on the test data: y_pred

y_pred = reg_model.predict(X_test)

reg_score = MSE(y_test, y_pred)

# Compute and print R^2 and RMSE

print("R^2: {}".format(reg.score))

rmse = np.sqrt(reg_score)

print("Root Mean Squared Error(RMSE-linear): {}".format(rmse))





#Verify Shape of Training Set

print(X_train.shape)

print(y_train.shape)





#Verify Shape of Training Set

print("X_test.shape", X_test.shape)

print("y_test.shape",y_test.shape)



# Instantiate dt

dt = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13,random_state=3)



# Fit dt to the training set

dt.fit(X_train, y_train)

# Compute y_pred

y_pred= dt.predict(X_test)



# Compute mse_dt

mse_dt = MSE(y_test, y_pred)



# Compute rmse_dt

rmse_dt = np.sqrt(mse_dt)



# Print rmse_dt

print("Test set RMSE of dt: {:.2f}".format(rmse_dt))



# Compute the array containing the 10-folds CV MSEs

MSE_CV10_scores = - cross_val_score(dt, X_train, y_train, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



# Compute the 10-folds CV RMSE

RMSE_CV10 = (MSE_CV10_scores.mean()**2)**(1/2)



# Print RMSE_CV

print('CV RMSE: {:.2f}'.format(RMSE_CV10))



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

 

# Create a linear regression object: reg

reg = LinearRegression()



# Perform 3-fold CV

cvscores_3 = cross_val_score(reg,X,y,cv=3)

mean_cv_reg= np.mean(cvscores_3)



# Perform 10-fold CV

cvscores_10 = cross_val_score(reg,X,y,cv=10)

print(np.mean(cvscores_10))





# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE



# Fit dt to the training set

dt.fit(X_train, y_train)



# Predict the labels of the training set

y_pred_dt = dt.predict(X_train)

MSE_dt = MSE(y_test, y_pred)

# Evaluate the training set RMSE of dt

RMSE_dt = MSE_dt**(1/2)



# Print RMSE_train

print('RMSE_dt: {:.2f}'.format(RMSE_dt))



X= metadata_df

batch_size = 45

centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]

n_clusters = len(centers)

X, labels_true = make_blobs(n_samples=3000, centers=centers, cluster_std=0.7)



# #############################################################################

# Compute clustering with Means



kmeans = KMeans(init='k-means++', n_clusters=3, n_init=10)

t0 = time.time()

kmeans.fit(X)

t_batch = time.time() - t0



# #############################################################################

# Compute clustering with MiniBatchKMeans



mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,

                      n_init=10, max_no_improvement=10, verbose=0)

t0 = time.time()

mbk.fit(X)

t_mini_batch = time.time() - t0



# #############################################################################

# Plot result



fig = plt.figure(figsize=(10, 5))

fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)

colors = ['#4EACC5', '#FF9C34', '#4E9A06']



# We want to have the same colors for the same cluster from the

# MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per

# closest one.

kmeans_cluster_centers = kmeans.cluster_centers_

order = pairwise_distances_argmin(kmeans.cluster_centers_,

                                  mbk.cluster_centers_)

mbk_means_cluster_centers = mbk.cluster_centers_[order]



kmeans_labels = pairwise_distances_argmin(X, kmeans_cluster_centers)

mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)



# KMeans

ax = fig.add_subplot(1, 3, 1)

for k, col in zip(range(n_clusters), colors):

    my_members = kmeans_labels == k

    cluster_center = kmeans_cluster_centers[k]

    ax.plot(X[my_members, 0], X[my_members, 1], 'w',

            markerfacecolor=col, marker='.')

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,

            markeredgecolor='k', markersize=6)

ax.set_title('KMeans')

ax.set_xticks(())

ax.set_yticks(())

plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (

    t_batch, kmeans.inertia_))



# MiniBatchKMeans

ax = fig.add_subplot(1, 3, 2)

for k, col in zip(range(n_clusters), colors):

    my_members = mbk_means_labels == k

    cluster_center = mbk_means_cluster_centers[k]

    ax.plot(X[my_members, 0], X[my_members, 1], 'w',

            markerfacecolor=col, marker='.')

    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,

            markeredgecolor='k', markersize=6)

ax.set_title('MiniBatchKMeans')

ax.set_xticks(())

ax.set_yticks(())

plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %

         (t_mini_batch, mbk.inertia_))



# Initialise the different array to all False

different = (mbk_means_labels == 4)

ax = fig.add_subplot(1, 3, 3)



for k in range(n_clusters):

    different += ((kmeans_labels == k) != (mbk_means_labels == k))



identic = np.logical_not(different)

ax.plot(X[identic, 0], X[identic, 1], 'w',

        markerfacecolor='#bbbbbb', marker='.')

ax.plot(X[different, 0], X[different, 1], 'w',

        markerfacecolor='m', marker='.')

ax.set_title('Difference')

ax.set_xticks(())

ax.set_yticks(())



plt.show()



# KMeans Cluster Analysis



ks = range(1, 10)

inertias = []



for k in ks:

    # Create a KMeans instance with k clusters: model

    kmeans = KMeans(n_clusters=k)

    

    # Fit model to samples

    kmeans_model= kmeans.fit(metadata_df)

    

      # Append the inertia to the list of inertias

    inertias.append(kmeans_model.inertia_)

    

# Plot ks vs inertias

plt.plot(ks, inertias, '-o')

plt.xlabel('number of clusters, k')

plt.ylabel('inertia')

plt.xticks(ks)

plt.show()


"""

============================================================================

Affinity propagation clustering algorithm, Adapted from sklearn

============================================================================



Reference:

Brendan J. Frey and Delbert Dueck, "Clustering by Passing Messages

Between Data Points", Science Feb. 2007



"""



from sklearn.cluster import AffinityPropagation

from sklearn import metrics

from sklearn.datasets import make_blobs

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.metrics.pairwise import pairwise_distances_argmin

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans



#Generate a synthetic dataset



rand = np.random.random((3000, 380))

print(rand)



rand.shape

metadata = rand

metadata



# Convert numpy array, rand into a Pandas dataframe



metadata_df = pd.DataFrame(rand)

metadata_df







# Separate the data into X and y components; 

#Define the y-component conforming to a trignonmetric function, np.tanh 

#X= metadata[:,]

#y = np.tanh(metadata[:, 0:380])



# #############################################################################

# Generate sample data



centers = [[1, 1], [-1, -1], [1, -1]]

X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,

                            random_state=0)



af = AffinityPropagation(preference=-50).fit(X)

cluster_centers_indices = af.cluster_centers_indices_

labels = af.labels_



n_clusters_ = len(cluster_centers_indices)



print('Estimated number of clusters: %d' % n_clusters_)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))

print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))

print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))

print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))

print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(labels_true, labels))

print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels, metric='euclidean'))



# #############################################################################

# Plot result

import matplotlib.pyplot as plt

from itertools import cycle



plt.close('all')

plt.figure(1)

plt.clf()



colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')

for k, col in zip(range(n_clusters_), colors):

    my_members = labels == k

    cluster_center = X[cluster_centers_indices[k]]

    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')

    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)

    for x in X[my_members]:

        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)



plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.show()
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline





rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe

metadata_df = pd.DataFrame(rand) 

metadata_df



# Import Normalizer

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



# Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=10)



# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)







# Fit pipeline to metadata



pipe_metadata_df = pipeline.fit_transform(metadata_df)

pipe_df= pipe_metadata_df

print(pipe_df.shape)



X= pipe_df

y= 1/X



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)





# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)







# Fit pipeline to metadata



pipe_metadata_df = pipeline.fit_transform(metadata_df)

pipe_df= pipe_metadata_df

print(pipe_df.shape)



X= pipe_df

y= 1/X



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)





# Create the regressor: reg_all

reg = LinearRegression()



# Fit the regressor to the training data

reg_model = reg.fit(X_train, y_train)



# Predict on the test data: y_pred

y_pred = reg.predict(X_test)



# Compute and print R^2 and RMSE



rmse = np.sqrt(MSE(y_test, y_pred))

print("Root Mean Squared Error (LINEAR REGRESSION): {}".format(rmse))
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline()

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline





rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe

metadata_df = pd.DataFrame(rand) 

metadata_df







# Import Normalizer

# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



# Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=10)



# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)







# Fit pipeline to metadata



pipe_metadata_df = pipeline.fit_transform(metadata_df)

pipe_df= pipe_metadata_df

print(pipe_df.shape)



X=pipe_df

y= 1/X



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)





# Now let's try running the RandomForest Regressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate RandomForestRegressor



rfr = RandomForestRegressor(n_estimators = 100,random_state=2)

            

# Fit rf to the training set    

rfr.fit(X_train, y_train) 



# Compute y_pred

y_pred= rfr.predict(X_test)



# Compute MSE for Test Set, mse_rfr_test

mse_rfr_test = MSE(y_test, y_pred)

print("MSE for RandomForestRegressor:", mse_rfr_test)

# Compute rmse_rfr

rmse_rfr_test = mse_rfr_test**(1/2)



#Cross Validate on Test set 



from sklearn.model_selection import cross_val_score



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_train = - cross_val_score(rfr, X_train, y_train, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Training set MSE_CV_Score",MSE_CV_scores_rfr_train)



# Compute the 10-folds CV RMSE

RMSE_CV_rfr_train = (MSE_CV_scores_rfr_train.mean())**(1/2)





# Print rmse_rfr

print("Training set RMSE of rfr: {:.2f}".format(RMSE_CV_rfr_train))



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_test = - cross_val_score(rfr, X_test, y_test, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Test set MSE_CV_Scores for rfr:",MSE_CV_scores_rfr_test)



# Compute the 10-folds CV RMSE on Test Set

RMSE_CV_rfr_test = (MSE_CV_scores_rfr_test.mean())**(1/2)





# Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for rfr:",RMSE_CV_rfr_test)
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline()

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline





rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe

metadata_df = pd.DataFrame(rand) 

metadata_df







# Import Normalizer

# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



# Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=10)



# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)







# Fit pipeline to metadata



pipe_metadata_df = pipeline.fit_transform(metadata_df)

pipe_df= pipe_metadata_df

print(pipe_df.shape)



X=pipe_df

y= 1/X



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)





# Now let's try running the RandomForest Regressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate RandomForestRegressor



rfr = RandomForestRegressor(n_estimators = 1000,random_state=2)

            

# Fit rf to the training set    

rfr.fit(X_train, y_train) 



# Compute y_pred

y_pred= rfr.predict(X_test)



# Compute MSE for Test Set, mse_rfr_test

mse_rfr_test = MSE(y_test, y_pred)

print("MSE for RandomForestRegressor:", mse_rfr_test)

# Compute rmse_rfr

rmse_rfr_test = mse_rfr_test**(1/2)



#Cross Validate on Test set 



from sklearn.model_selection import cross_val_score



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_train = - cross_val_score(rfr, X_train, y_train, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Training set MSE_CV_Score",MSE_CV_scores_rfr_train)



# Compute the 10-folds CV RMSE

RMSE_CV_rfr_train = (MSE_CV_scores_rfr_train.mean())**(1/2)





# Print rmse_rfr

print("Training set RMSE of rfr: {:.2f}".format(RMSE_CV_rfr_train))



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_test = - cross_val_score(rfr, X_test, y_test, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Test set MSE_CV_Scores for rfr:",MSE_CV_scores_rfr_test)



# Compute the 10-folds CV RMSE on Test Set

RMSE_CV_rfr_test = (MSE_CV_scores_rfr_test.mean())**(1/2)





# Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for rfr:",RMSE_CV_rfr_test)
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline()

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline





rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe

metadata_df = pd.DataFrame(rand) 

metadata_df







# Import Normalizer

# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



# Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=10)



# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)







# Fit pipeline to metadata



pipe_metadata_df = pipeline.fit_transform(metadata_df)

pipe_df= pipe_metadata_df

print(pipe_df.shape)



X=pipe_df

y= 1/X



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)





# Now let's try running the RandomForest Regressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate RandomForestRegressor



rfr = RandomForestRegressor(n_estimators = 1000, max_depth=10, random_state=2)

            

# Fit rf to the training set    

rfr.fit(X_train, y_train) 



# Compute y_pred

y_pred= rfr.predict(X_test)



# Compute MSE for Test Set, mse_rfr_test

mse_rfr_test = MSE(y_test, y_pred)

print("MSE for RandomForestRegressor:", mse_rfr_test)



# Compute rmse_rfr

rmse_rfr_test = mse_rfr_test**(1/2)



#Cross Validate on Test set 



from sklearn.model_selection import cross_val_score



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_train = - cross_val_score(rfr, X_train, y_train, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Training set MSE_CV_Score",MSE_CV_scores_rfr_train)



# Compute the 10-folds CV RMSE

RMSE_CV_rfr_train = (MSE_CV_scores_rfr_train.mean())**(1/2)





# Print rmse_rfr

print("Training set RMSE of rfr: {:.2f}".format(RMSE_CV_rfr_train))



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_test = - cross_val_score(rfr, X_test, y_test, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Test set MSE_CV_Scores for rfr:",MSE_CV_scores_rfr_test)



# Compute the 10-folds CV RMSE on Test Set

RMSE_CV_rfr_test = (MSE_CV_scores_rfr_test.mean())**(1/2)





# Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for rfr:",RMSE_CV_rfr_test)
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline()

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline





rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe

metadata_df = pd.DataFrame(rand) 

metadata_df







# Import Normalizer

# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



# Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=10)



# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)







# Fit pipeline to metadata



pipe_metadata_df = pipeline.fit_transform(metadata_df)

pipe_df= pipe_metadata_df

print(pipe_df.shape)



X=pipe_df

y= 1/X





# Split the dataset, using train_test_split



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)



# Now let's try running the RandomForest Regressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate RandomForestRegressor



rfr = RandomForestRegressor(n_estimators = 1000, max_depth=10, random_state=2)

            

# Fit rf to the training set    

rfr.fit(X_train, y_train) 



# Compute y_pred

y_pred= rfr.predict(X_test)



# Compute MSE for Test Set, mse_rfr_test

mse_rfr_test = MSE(y_test, y_pred)

print("MSE for RandomForestRegressor:", mse_rfr_test)



# Compute rmse_rfr

rmse_rfr_test = mse_rfr_test**(1/2)



#Cross Validate on Test set 



from sklearn.model_selection import cross_val_score



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_train = - cross_val_score(rfr, X_train, y_train, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Training set MSE_CV_Scores (rfr-1000):",MSE_CV_scores_rfr_train)



# Compute the 10-folds CV RMSE

RMSE_CV_rfr_train = (MSE_CV_scores_rfr_train.mean())**(1/2)





# Print rmse_rfr

print("Training set RMSE (rfr-1000): {:.2f}".format(RMSE_CV_rfr_train))



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_test = - cross_val_score(rfr, X_test, y_test, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Test set MSE_CV_Scores for rfr:", MSE_CV_scores_rfr_test)



# Compute the 10-folds CV RMSE on Test Set

RMSE_CV_rfr_test = (MSE_CV_scores_rfr_test.mean())**(1/2)





# Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for rfr:",RMSE_CV_rfr_test)







#Instantiate GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3, 10],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

# Create a base model

rfr = RandomForestRegressor()

# Instantiate the grid search modelt

gs_cv = GridSearchCV(estimator = rfr, param_grid = param_grid, 

                          cv = 5, n_jobs = -1, verbose = 2)



gs_cv_model = gs_cv.fit(X_train, y_train)

y_pred = gs_cv_model.predict(X_test)

gscv_best_grid = gs_cv_model.best_estimator_

gscv_best_params = gs_cv_model.best_params_

print("BEST ESTIMATOR --> (gscv_best_grid->estimator):",gscv_best_grid)

print("BEST PARAMS --> (gscv_best_params):",gscv_best_params )
import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import plot

from sklearn import feature_extraction

from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import xgboost as xgb

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics.pairwise import pairwise_distances_argmin

import seaborn as sns

from sklearn.datasets import make_blobs

from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, train_test_split

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline()

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline





rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe

metadata_df = pd.DataFrame(rand) 

metadata_df







# Import Normalizer

# Import mean_squared_error from sklearn.metrics as MSE

from sklearn.metrics import mean_squared_error as MSE

from sklearn.preprocessing import Normalizer

from sklearn.pipeline import make_pipeline



# Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=10)



# Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)







# Fit pipeline to metadata



pipe_metadata_df = pipeline.fit_transform(metadata_df)

pipe_df= pipe_metadata_df

print(pipe_df.shape)



X=pipe_df

y= 1/X





# Split the dataset, using train_test_split



# Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)



# Now let's try running the RandomForest Regressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate RandomForestRegressor



rfr = RandomForestRegressor(n_estimators = 1000, max_depth=10, random_state=2)

            

# Fit rf to the training set    

rfr.fit(X_train, y_train) 



# Compute y_pred

y_pred= rfr.predict(X_test)



# Compute MSE for Test Set, mse_rfr_test

mse_rfr_test = MSE(y_test, y_pred)

print("MSE for RandomForestRegressor:", mse_rfr_test)



# Compute rmse_rfr

rmse_rfr_test = mse_rfr_test**(1/2)



#Cross Validate on Test set 



from sklearn.model_selection import cross_val_score



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_train = - cross_val_score(rfr, X_train, y_train, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Training set MSE_CV_Scores (rfr-1000):",MSE_CV_scores_rfr_train)



# Compute the 10-folds CV RMSE

RMSE_CV_rfr_train = (MSE_CV_scores_rfr_train.mean())**(1/2)





# Print rmse_rfr

print("Training set RMSE (rfr-1000): {:.2f}".format(RMSE_CV_rfr_train))



# Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_rfr_test = - cross_val_score(rfr, X_test, y_test, cv=10, 

                       scoring='neg_mean_squared_error',

                       n_jobs=-1)



print("Test set MSE_CV_Scores for rfr:", MSE_CV_scores_rfr_test)



# Compute the 10-folds CV RMSE on Test Set

RMSE_CV_rfr_test = (MSE_CV_scores_rfr_test.mean())**(1/2)





# Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for rfr:",RMSE_CV_rfr_test)







#Instantiate GridSearchCV

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3, 10],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

# Create a based model

rfr = RandomForestRegressor()

# Instantiate the grid search modelt

gs_cv = GridSearchCV(estimator = rfr, param_grid = param_grid, 

                          cv = 5, n_jobs = -1, verbose = 2)



gs_cv_model = gs_cv.fit(X_train, y_train)

y_pred = gs_cv_model.predict(X_test)

gscv_best_grid = gs_cv_model.best_estimator_

gscv_best_params = gs_cv_model.best_params_

print("BEST ESTIMATOR --> (gscv_best_grid->estimator):",gscv_best_grid)

print("BEST PARAMS --> (gscv_best_params):",gscv_best_params )
import time 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import plot 

from sklearn import feature_extraction 

from sklearn.linear_model import LinearRegression, ElasticNet 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

import xgboost as xgb 

from sklearn.metrics.pairwise import cosine_similarity 

from sklearn.metrics.pairwise import pairwise_distances_argmin 

import seaborn as sns 

from sklearn.datasets import make_blobs 

from sklearn.cluster import MiniBatchKMeans, KMeans 

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor 

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV, train_test_split 

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline() 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe 

metadata_df = pd.DataFrame(rand) 

metadata_df



#Import Normalizer

from sklearn.metrics import mean_squared_error as MSE 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



#Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans kmeans = KMeans(n_clusters=10)

kmeans = KMeans(n_clusters=10)

#Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)



#Fit pipeline to metadata

pipe_metadata_df = pipeline.fit_transform(metadata_df) 

pipe_df= pipe_metadata_df 



print(pipe_df.shape)

X=metadata_df

y=np.sin(metadata_df)



#X_train,X_test = metadata_df[:3000], metadata_df[3000:] 

#y_train, y_test = np.sin(metadata_df[:3000]), np.sin(metadata_df[3000:])



#Prepare for GradientBoostingRegressor

#Make Regression

from sklearn.datasets import make_regression



X,y = make_regression(n_samples = 3000, n_features = 10,random_state=231)



#Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=231)



#Now let's try running the GradientBoostingRegressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate GradientBoostingRegressor



gb = GradientBoostingRegressor(n_estimators = 10000, max_depth = 80, max_features= 10, random_state=231)



#Fit rf to the training set

gb.fit(X_train, y_train)



#Compute y_pred using X_test

y_pred= gb.predict(X_test)



#Compute MSE for Test Set, mse_rfr_test

mse_gb_test = MSE(y_test, y_pred) 

print("MSE for GradientBoostingRegressor (GB):p", mse_gb_test)



#Compute rmse_rfr

rmse_gb_test = mse_gb_test**(1/2)



#Cross Validate on Test set



from sklearn.model_selection import cross_val_score



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_gb_train = - cross_val_score(gb, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)



print("Training set MSE_CV_Scores (gb-1000):",MSE_CV_scores_gb_train)



#Compute the 10-folds CV RMSE

RMSE_CV_gb_train = (MSE_CV_scores_gb_train.mean())**(1/2)



#Print rmse_gb

print("Training set RMSE (gb-1000): {:.2f}".format(RMSE_CV_gb_train))



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_gb_test = - cross_val_score(gb, X_test, y_test, cv=10,scoring='neg_mean_squared_error', n_jobs=-1)



print("Test set MSE_CV_Scores for gb:", MSE_CV_scores_gb_test)



#Compute the 10-folds CV RMSE on Test Set

RMSE_CV_gb_test = (MSE_CV_scores_gb_test.mean())**(1/2)



#Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for gb:",RMSE_CV_gb_test)



#Instantiate GridSearchCV



#Create the parameter grid based on the results of random search

param_grid = {'max_depth': [80, 90, 100, 110], 

              'max_features': [2, 3, 10], 

              'min_samples_leaf': [3, 4, 5], 

              'min_samples_split': [8, 10, 12], 

              'n_estimators': [100, 200, 300, 1000]}



#Create a based model

gb = GradientBoostingRegressor()



#Instantiate the grid search modelt

gs_cv = GridSearchCV(estimator = gb, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)



gs_cv_model = gs_cv.fit(X_train, y_train) 

y_pred = gs_cv_model.predict(X_test) 

gscv_best_grid = gs_cv_model.best_estimator_ 

gscv_best_params = gs_cv_model.best_params_ 

print("BEST ESTIMATOR --> (gscv_best_grid->estimator):",gscv_best_grid) 

print("BEST PARAMS --> (gscv_best_params):",gscv_best_params )
from collections import defaultdict



import time

import gc

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.utils import shuffle





def _not_in_sphinx():

    # Hack to detect whether we are running by the sphinx builder

    return '__file__' in globals()





def atomic_benchmark_estimator(estimator, X_test, verbose=False):

    """Measure runtime prediction of each instance."""

    n_instances = X_test.shape[0]

    runtimes = np.zeros(n_instances, dtype=np.float)

    for i in range(n_instances):

        instance = X_test[[i], :]

        start = time.time()

        estimator.predict(instance)

        runtimes[i] = time.time() - start

    if verbose:

        print("atomic_benchmark runtimes:", min(runtimes), np.percentile(

            runtimes, 50), max(runtimes))

    return runtimes





def bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats, verbose):

    """Measure runtime prediction of the whole input."""

    n_instances = X_test.shape[0]

    runtimes = np.zeros(n_bulk_repeats, dtype=np.float)

    for i in range(n_bulk_repeats):

        start = time.time()

        estimator.predict(X_test)

        runtimes[i] = time.time() - start

    runtimes = np.array(list(map(lambda x: x / float(n_instances), runtimes)))

    if verbose:

        print("bulk_benchmark runtimes:", min(runtimes), np.percentile(

            runtimes, 50), max(runtimes))

    return runtimes





def benchmark_estimator(estimator, X_test, n_bulk_repeats=30, verbose=False):

    """

    Measure runtimes of prediction in both atomic and bulk mode.



    Parameters

    ----------

    estimator : already trained estimator supporting `predict()`

    X_test : test input

    n_bulk_repeats : how many times to repeat when evaluating bulk mode



    Returns

    -------

    atomic_runtimes, bulk_runtimes : a pair of `np.array` which contain the

    runtimes in seconds.



    """

    atomic_runtimes = atomic_benchmark_estimator(estimator, X_test, verbose)

    bulk_runtimes = bulk_benchmark_estimator(estimator, X_test, n_bulk_repeats,

                                             verbose)

    return atomic_runtimes, bulk_runtimes





def generate_dataset(n_train, n_test, n_features, noise=0.1, verbose=False):

    """Generate a regression dataset with the given parameters."""

    if verbose:

        print("generating dataset...")



    X, y, coef = make_regression(n_samples=n_train + n_test,

                                 n_features=n_features, noise=noise, coef=True)



    random_seed = 13

    X_train, X_test, y_train, y_test = train_test_split(

        X, y, train_size=n_train, test_size=n_test, random_state=random_seed)

    X_train, y_train = shuffle(X_train, y_train, random_state=random_seed)



    X_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)

    X_test = X_scaler.transform(X_test)



    y_scaler = StandardScaler()

    y_train = y_scaler.fit_transform(y_train[:, None])[:, 0]

    y_test = y_scaler.transform(y_test[:, None])[:, 0]



    gc.collect()

    if verbose:

        print("ok")

    return X_train, y_train, X_test, y_test





def boxplot_runtimes(runtimes, pred_type, configuration):

    """

    Plot a new `Figure` with boxplots of prediction runtimes.



    Parameters

    ----------

    runtimes : list of `np.array` of latencies in micro-seconds

    cls_names : list of estimator class names that generated the runtimes

    pred_type : 'bulk' or 'atomic'



    """



    fig, ax1 = plt.subplots(figsize=(10, 6))

    bp = plt.boxplot(runtimes, )



    cls_infos = ['%s\n(%d %s)' % (estimator_conf['name'],

                                  estimator_conf['complexity_computer'](

                                      estimator_conf['instance']),

                                  estimator_conf['complexity_label']) for

                 estimator_conf in configuration['estimators']]

    plt.setp(ax1, xticklabels=cls_infos)

    plt.setp(bp['boxes'], color='black')

    plt.setp(bp['whiskers'], color='black')

    plt.setp(bp['fliers'], color='red', marker='+')



    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',

                   alpha=0.5)



    ax1.set_axisbelow(True)

    ax1.set_title('Prediction Time per Instance - %s, %d feats.' % (

        pred_type.capitalize(),

        configuration['n_features']))

    ax1.set_ylabel('Prediction Time (us)')



    plt.show()





def benchmark(configuration):

    """Run the whole benchmark."""

    X_train, y_train, X_test, y_test = generate_dataset(

        configuration['n_train'], configuration['n_test'],

        configuration['n_features'])



    stats = {}

    for estimator_conf in configuration['estimators']:

        print("Benchmarking", estimator_conf['instance'])

        estimator_conf['instance'].fit(X_train, y_train)

        gc.collect()

        a, b = benchmark_estimator(estimator_conf['instance'], X_test)

        stats[estimator_conf['name']] = {'atomic': a, 'bulk': b}



    cls_names = [estimator_conf['name'] for estimator_conf in configuration[

        'estimators']]

    runtimes = [1e6 * stats[clf_name]['atomic'] for clf_name in cls_names]

    boxplot_runtimes(runtimes, 'atomic', configuration)

    runtimes = [1e6 * stats[clf_name]['bulk'] for clf_name in cls_names]

    boxplot_runtimes(runtimes, 'bulk (%d)' % configuration['n_test'],

                     configuration)





def n_feature_influence(estimators, n_train, n_test, n_features, percentile):

    """

    Estimate influence of the number of features on prediction time.



    Parameters

    ----------



    estimators : dict of (name (str), estimator) to benchmark

    n_train : nber of training instances (int)

    n_test : nber of testing instances (int)

    n_features : list of feature-space dimensionality to test (int)

    percentile : percentile at which to measure the speed (int [0-100])



    Returns:

    --------



    percentiles : dict(estimator_name,

                       dict(n_features, percentile_perf_in_us))



    """

    percentiles = defaultdict(defaultdict)

    for n in n_features:

        print("benchmarking with %d features" % n)

        X_train, y_train, X_test, y_test = generate_dataset(n_train, n_test, n)

        for cls_name, estimator in estimators.items():

            estimator.fit(X_train, y_train)

            gc.collect()

            runtimes = bulk_benchmark_estimator(estimator, X_test, 30, False)

            percentiles[cls_name][n] = 1e6 * np.percentile(runtimes,

                                                           percentile)

    return percentiles





def plot_n_features_influence(percentiles, percentile):

    fig, ax1 = plt.subplots(figsize=(10, 6))

    colors = ['r', 'g', 'b']

    for i, cls_name in enumerate(percentiles.keys()):

        x = np.array(sorted([n for n in percentiles[cls_name].keys()]))

        y = np.array([percentiles[cls_name][n] for n in x])

        plt.plot(x, y, color=colors[i], )

    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',

                   alpha=0.5)

    ax1.set_axisbelow(True)

    ax1.set_title('Evolution of Prediction Time with #Features')

    ax1.set_xlabel('#Features')

    ax1.set_ylabel('Prediction Time at %d%%-ile (us)' % percentile)

    plt.show()





def benchmark_throughputs(configuration, duration_secs=0.1):

    """benchmark throughput for different estimators."""

    X_train, y_train, X_test, y_test = generate_dataset(

        configuration['n_train'], configuration['n_test'],

        configuration['n_features'])

    throughputs = dict()

    for estimator_config in configuration['estimators']:

        estimator_config['instance'].fit(X_train, y_train)

        start_time = time.time()

        n_predictions = 0

        while (time.time() - start_time) < duration_secs:

            estimator_config['instance'].predict(X_test[[0]])

            n_predictions += 1

        throughputs[estimator_config['name']] = n_predictions / duration_secs

    return throughputs





def plot_benchmark_throughput(throughputs, configuration):

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['r', 'g', 'b']

    cls_infos = ['%s\n(%d %s)' % (estimator_conf['name'],

                                  estimator_conf['complexity_computer'](

                                      estimator_conf['instance']),

                                  estimator_conf['complexity_label']) for

                 estimator_conf in configuration['estimators']]

    cls_values = [throughputs[estimator_conf['name']] for estimator_conf in

                  configuration['estimators']]

    plt.bar(range(len(throughputs)), cls_values, width=0.5, color=colors)

    ax.set_xticks(np.linspace(0.25, len(throughputs) - 0.75, len(throughputs)))

    ax.set_xticklabels(cls_infos, fontsize=10)

    ymax = max(cls_values) * 1.2

    ax.set_ylim((0, ymax))

    ax.set_ylabel('Throughput (predictions/sec)')

    ax.set_title('Prediction Throughput for different estimators (%d '

                 'features)' % configuration['n_features'])

    plt.show()





# #############################################################################

# Main code



start_time = time.time()



# #############################################################################

# Benchmark bulk/atomic prediction speed for various regressors

configuration = {

    'n_train': int(1e3),

    'n_test': int(1e2),

    'n_features': int(1e2),

    'estimators': [

        {'name': 'Linear Model',

         'instance': SGDRegressor(penalty='elasticnet', alpha=0.01,

                                  l1_ratio=0.25, tol=1e-4),

         'complexity_label': 'non-zero coefficients',

         'complexity_computer': lambda clf: np.count_nonzero(clf.coef_)},

        {'name': 'RandomForest',

         'instance': RandomForestRegressor(),

         'complexity_label': 'estimators',

         'complexity_computer': lambda clf: clf.n_estimators},

        {'name': 'SVR',

         'instance': SVR(kernel='rbf'),

         'complexity_label': 'support vectors',

         'complexity_computer': lambda clf: len(clf.support_vectors_)},

    ]

}

benchmark(configuration)



# benchmark n_features influence on prediction speed

percentile = 90

percentiles = n_feature_influence({'ridge': Ridge()},

                                  configuration['n_train'],

                                  configuration['n_test'],

                                  [100, 250, 500], percentile)

plot_n_features_influence(percentiles, percentile)



# benchmark throughput

throughputs = benchmark_throughputs(configuration)

plot_benchmark_throughput(throughputs, configuration)



stop_time = time.time()

print("example run in %.2fs" % (stop_time - start_time))



import time 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import plot 

from sklearn import feature_extraction 

from sklearn.linear_model import LinearRegression, ElasticNet 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

import xgboost as xgb 

from sklearn.metrics.pairwise import cosine_similarity 

from sklearn.metrics.pairwise import pairwise_distances_argmin 

import seaborn as sns 

from sklearn.datasets import make_blobs 

from sklearn.cluster import MiniBatchKMeans, KMeans 

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor 

from sklearn.svm import SVR 

from sklearn.model_selection import GridSearchCV, train_test_split 

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline() 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe 

metadata_df = pd.DataFrame(rand) 

metadata_df



#Import Normalizer

from sklearn.metrics import mean_squared_error as MSE 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



#Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans kmeans = KMeans(n_clusters=10)

kmeans = KMeans(n_clusters=10)

#Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)



#Fit pipeline to metadata

pipe_metadata_df = pipeline.fit_transform(metadata_df) 

pipe_df= pipe_metadata_df 



print(pipe_df.shape)

X=metadata_df

y=np.sin(metadata_df)



#X_train,X_test = metadata_df[:3000], metadata_df[3000:] 

#y_train, y_test = np.sin(metadata_df[:3000]), np.sin(metadata_df[3000:])



#Prepare for StochasticGradientDescentRegressor

#Make Regression



from collections import defaultdict

import time

import gc

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.utils import shuffle

from sklearn.datasets import make_regression



X,y = make_regression(n_samples = 3000, n_features = 10,random_state=231)



#Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=231)



#Now let's try running the GradientBoostingRegressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate GradientBoostingRegressor



sgd = SGDRegressor(random_state=231)



#Fit rf to the training set

sgd.fit(X_train, y_train)



#Compute y_pred using X_test

y_pred= sgd.predict(X_test)



#Compute MSE for Test Set, mse_rfr_test

mse_sgd_test = MSE(y_test, y_pred) 

print("Stochasic Gradient Descent Regressor (SGDRegresor):", mse_sgd_test)



#Compute rmse_rfr

rmse_sgd_test = mse_sgd_test**(1/2)



#Cross Validate on Test set



from sklearn.model_selection import cross_val_score



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_train = - cross_val_score(sgd, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)



print("Training set MSE_CV_Scores (sgd-1000):",MSE_CV_scores_sgd_train)



#Compute the 10-folds CV RMSE

RMSE_CV_sgd_train = (MSE_CV_scores_sgd_train.mean())**(1/2)



#Print rmse_sgd

print("Training set RMSE (sgd-1000): {:.2f}".format(RMSE_CV_sgd_train))



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_test = - cross_val_score(sgd, X_test, y_test, cv=10,scoring='neg_mean_squared_error', n_jobs=-1)



print("Test set MSE_CV_Scores for sgd:", MSE_CV_scores_sgd_test)



#Compute the 10-folds CV RMSE on Test Set

RMSE_CV_sgd_test = (MSE_CV_scores_sgd_test.mean())**(1/2)



#Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for sgd:",RMSE_CV_sgd_test)



#Instantiate GridSearchCV

import time 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import plot 

from sklearn import feature_extraction 

from sklearn.linear_model import LinearRegression, ElasticNet 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

import xgboost as xgb 

from sklearn.metrics.pairwise import cosine_similarity 

from sklearn.metrics.pairwise import pairwise_distances_argmin 

import seaborn as sns 

from sklearn.datasets import make_blobs 

from sklearn.cluster import MiniBatchKMeans, KMeans 

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor 

from sklearn.svm import SVR 

from sklearn.model_selection import GridSearchCV, train_test_split 

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline() 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe 

metadata_df = pd.DataFrame(rand) 

metadata_df



#Import Normalizer

from sklearn.metrics import mean_squared_error as MSE 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



#Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans kmeans = KMeans(n_clusters=10)

kmeans = KMeans(n_clusters=10)

#Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)



#Fit pipeline to metadata

pipe_metadata_df = pipeline.fit_transform(metadata_df) 

pipe_df= pipe_metadata_df 



print(pipe_df.shape)

X=metadata_df

y=np.sin(metadata_df)



#X_train,X_test = metadata_df[:3000], metadata_df[3000:] 

#y_train, y_test = np.sin(metadata_df[:3000]), np.sin(metadata_df[3000:])



#Prepare for StochasticGradientDescentRegressor

#Make Regression



from collections import defaultdict

import time

import gc

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.utils import shuffle

from sklearn.datasets import make_regression



X,y = make_regression(n_samples = 3000, n_features = 10,random_state=231)



#Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=231)



#Now let's try running the GradientBoostingRegressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate GradientBoostingRegressor



sgd = SGDRegressor(random_state=231)



#Fit rf to the training set

sgd.fit(X_train, y_train)



#Compute y_pred using X_test

y_pred= sgd.predict(X_test)



#Compute MSE for Test Set, mse_rfr_test

mse_sgd_test = MSE(y_test, y_pred) 

print("Stochastic SGDRegressor:", mse_sgd_test)



#Compute rmse_rfr

rmse_sgd_test = mse_sgd_test**(1/2)



#Cross Validate on Test set



from sklearn.model_selection import cross_val_score



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_train = - cross_val_score(sgd, X_train, y_train, 

                                            cv=10, scoring='neg_mean_squared_error', n_jobs=-1)



print("Training set MSE_CV_Scores (sgd-1000):",MSE_CV_scores_sgd_train)



#Compute the 10-folds CV RMSE

RMSE_CV_sgd_train = (MSE_CV_scores_sgd_train.mean())**(1/2)



#Print rmse_sgd

print("Training set RMSE (sgd-1000): {:.2f}".format(RMSE_CV_sgd_train))



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_test = - cross_val_score(sgd, X_test, y_test, cv=10,scoring='neg_mean_squared_error', n_jobs=-1)



print("Test set MSE_CV_Scores for sgd:", MSE_CV_scores_sgd_test)



#Compute the 10-folds CV RMSE on Test Set

RMSE_CV_sgd_test = (MSE_CV_scores_sgd_test.mean())**(1/2)



#Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for sgd:",RMSE_CV_sgd_test)



#Instantiate GridSearchCV



#Create the parameter grid based on the results of random search

param_grid = {'loss': ['squared_loss','huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 

              'penalty' : ['l2', 'l1', 'elasticnet'], 

              'alpha' : [0.0001, 0.0010, 0.01], 

              'fit_intercept' : [True, False], 

              'max_iter': [100,1000, 10000, 100000], 

              'epsilon' : [0.01, 0.1,0.22]}



#Create a based model

sgd = SGDRegressor()



#Instantiate the grid search modelt

gs_cv = GridSearchCV(estimator = sgd, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)

gs_cv_model = gs_cv.fit(X_train, y_train) 

y_pred = gs_cv_model.predict(X_test) 

gscv_best_grid = gs_cv_model.best_estimator_ 

gscv_best_params = gs_cv_model.best_params_ 

print("BEST ESTIMATOR --> (gscv_best_grid->estimator):",gscv_best_grid) 

print("BEST PARAMS --> (gscv_best_params):",gscv_best_params )       





#Prepare 
import time 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import plot 

from sklearn import feature_extraction 

from sklearn.linear_model import LinearRegression, ElasticNet 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

import xgboost as xgb 

from sklearn.metrics.pairwise import cosine_similarity 

from sklearn.metrics.pairwise import pairwise_distances_argmin 

import seaborn as sns 

from sklearn.datasets import make_blobs 

from sklearn.cluster import MiniBatchKMeans, KMeans 

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor 

from sklearn.svm import SVR 

from sklearn.model_selection import GridSearchCV, train_test_split 

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline() 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe 

metadata_df = pd.DataFrame(rand) 

metadata_df



#Import Normalizer

from sklearn.metrics import mean_squared_error as MSE 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



#Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans kmeans = KMeans(n_clusters=10)

kmeans = KMeans(n_clusters=10)

#Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)



#Fit pipeline to metadata

pipe_metadata_df = pipeline.fit_transform(metadata_df) 

pipe_df= pipe_metadata_df 



print(pipe_df.shape)

X=metadata_df

y=np.sin(metadata_df)



#X_train,X_test = metadata_df[:3000], metadata_df[3000:] 

#y_train, y_test = np.sin(metadata_df[:3000]), np.sin(metadata_df[3000:])



#Prepare for StochasticGradientDescentRegressor

#Make Regression



from collections import defaultdict

import time

import gc

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.utils import shuffle

from sklearn.datasets import make_regression



X,y = make_regression(n_samples = 3000, n_features = 10,random_state=231)



#Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=231)



#Now let's try running the GradientBoostingRegressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate GradientBoostingRegressor



sgd = SGDRegressor(random_state=231)



#Fit rf to the training set

sgd.fit(X_train, y_train)



#Compute y_pred using X_test

y_pred= sgd.predict(X_test)



#Compute MSE for Test Set, mse_rfr_test

mse_sgd_test = MSE(y_test, y_pred) 

print("Stochasic Gradient Descent Regressor (SGDRegresor):", mse_sgd_test)



#Compute rmse_rfr

rmse_sgd_test = mse_sgd_test**(1/2)



#Cross Validate on Test set



from sklearn.model_selection import cross_val_score



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_train = - cross_val_score(sgd, X_train, y_train, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)



print("Training set MSE_CV_Scores (sgd-1000):",MSE_CV_scores_sgd_train)



#Compute the 10-folds CV RMSE

RMSE_CV_sgd_train = (MSE_CV_scores_sgd_train.mean())**(1/2)



#Print rmse_sgd

print("Training set RMSE (sgd-1000): {:.2f}".format(RMSE_CV_sgd_train))



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_test = - cross_val_score(sgd, X_test, y_test, cv=10,scoring='neg_mean_squared_error', n_jobs=-1)



print("Test set MSE_CV_Scores for sgd:", MSE_CV_scores_sgd_test)



#Compute the 10-folds CV RMSE on Test Set

RMSE_CV_sgd_test = (MSE_CV_scores_sgd_test.mean())**(1/2)



#Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for sgd:",RMSE_CV_sgd_test)



#Instantiate GridSearchCV

import time 

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

from matplotlib.pyplot import plot 

from sklearn import feature_extraction 

from sklearn.linear_model import LinearRegression, ElasticNet 

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor 

import xgboost as xgb 

from sklearn.metrics.pairwise import cosine_similarity 

from sklearn.metrics.pairwise import pairwise_distances_argmin 

import seaborn as sns 

from sklearn.datasets import make_blobs 

from sklearn.cluster import MiniBatchKMeans, KMeans 

from sklearn.linear_model import LassoLars, RidgeCV, SGDRegressor 

from sklearn.svm import SVR 

from sklearn.model_selection import GridSearchCV, train_test_split 

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score



#Import Normalizer() and make_pipeline() 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



rand = np.random.random((3000, 380)) 

print(rand)



rand.shape 

metadata = rand 

metadata



#Convert numpy array, rand into a Pandas dataframe 

metadata_df = pd.DataFrame(rand) 

metadata_df



#Import Normalizer

from sklearn.metrics import mean_squared_error as MSE 

from sklearn.preprocessing import Normalizer 

from sklearn.pipeline import make_pipeline



#Create a normalizer: normalizer

normalizer = Normalizer()



#Create a KMeans model with 10 clusters: kmeans kmeans = KMeans(n_clusters=10)

kmeans = KMeans(n_clusters=10)

#Make a pipeline chaining normalizer and kmeans: pipeline

pipeline = make_pipeline(normalizer, kmeans)



#Fit pipeline to metadata

pipe_metadata_df = pipeline.fit_transform(metadata_df) 

pipe_df= pipe_metadata_df 



print(pipe_df.shape)

X=metadata_df

y=np.sin(metadata_df)



#X_train,X_test = metadata_df[:3000], metadata_df[3000:] 

#y_train, y_test = np.sin(metadata_df[:3000]), np.sin(metadata_df[3000:])



#Prepare for StochasticGradientDescentRegressor

#Make Regression



from collections import defaultdict

import time

import gc

import numpy as np

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.datasets import make_regression

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDRegressor

from sklearn.svm import SVR

from sklearn.utils import shuffle

from sklearn.datasets import make_regression



X,y = make_regression(n_samples = 3000, n_features = 10,random_state=231)



#Create training and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=231)



#Now let's try running the GradientBoostingRegressor on the piped normalized df (pipe_df) after running KMeans

#Instantiate GradientBoostingRegressor



sgd = SGDRegressor(random_state=231)



#Fit rf to the training set

sgd.fit(X_train, y_train)



#Compute y_pred using X_test

y_pred= sgd.predict(X_test)



#Compute MSE for Test Set, mse_rfr_test

mse_sgd_test = MSE(y_test, y_pred) 

print("Stochastic SGDRegressor:", mse_sgd_test)



#Compute rmse_rfr

rmse_sgd_test = mse_sgd_test**(1/2)



#Cross Validate on Test set



from sklearn.model_selection import cross_val_score



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_train = - cross_val_score(sgd, X_train, y_train, 

                                            cv=10, scoring='neg_mean_squared_error', n_jobs=-1)



print("Training set MSE_CV_Scores (sgd-1000):",MSE_CV_scores_sgd_train)



#Compute the 10-folds CV RMSE

RMSE_CV_sgd_train = (MSE_CV_scores_sgd_train.mean())**(1/2)



#Print rmse_sgd

print("Training set RMSE (sgd-1000): {:.2f}".format(RMSE_CV_sgd_train))



#Compute the array containing the 10-folds CV MSEs

MSE_CV_scores_sgd_test = - cross_val_score(sgd, X_test, y_test, cv=10,scoring='neg_mean_squared_error', n_jobs=-1)



print("Test set MSE_CV_Scores for sgd:", MSE_CV_scores_sgd_test)



#Compute the 10-folds CV RMSE on Test Set

RMSE_CV_sgd_test = (MSE_CV_scores_sgd_test.mean())**(1/2)



#Print rmse_cv_rfr_test

print("Test set RMSE_CV_Score for sgd:",RMSE_CV_sgd_test)



#Instantiate GridSearchCV



#Create the parameter grid based on the results of random search

param_grid = {'loss': ['squared_loss','huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 

              'penalty' : ['l2', 'l1', 'elasticnet'], 

              'alpha' : [0.0001, 0.0010, 0.01], 

              'fit_intercept' : [True, False], 

              'max_iter': [100,1000, 10000, 100000], 

              'epsilon' : [0.01, 0.1,0.22]}



#Create a based model

sgd = SGDRegressor()



#Instantiate the grid search modelt

gs_cv = GridSearchCV(estimator = sgd, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2)

gs_cv_model = gs_cv.fit(X_train, y_train) 

y_pred = gs_cv_model.predict(X_test) 

gscv_best_grid = gs_cv_model.best_estimator_ 

gscv_best_params = gs_cv_model.best_params_ 

print("BEST ESTIMATOR --> (gscv_best_grid->estimator):",gscv_best_grid) 

print("BEST PARAMS --> (gscv_best_params):",gscv_best_params )       



#Prepare for t-SNE using data pipe made with Kmeans, PCA, i.e. pipe_df

#Instantiate TSNE

import plotly.express as px

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE



Xt = pipe_df

yt = np.sin(Xt)

pca= PCA(n_components = 10)

pca_pipe_df = pca.fit_transform(Xt,yt)



pca = PCA(n_components=10)

pca.fit(metadata_df)

exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)

print(exp_var_cumul)



px.area(

    x=range(1, exp_var_cumul.shape[0] + 1),

    y=exp_var_cumul,

    labels={"x": "# Components", "y": "Explained Variance"}

)
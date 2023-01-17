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
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

%config InlineBackend.figure_format = 'retina'
X0 = pd.read_csv(r"/kaggle/input/comp-hardware-performance/machine.data.txt", header=None)
X0.head()
X0.rename({0:"Vendor",1:"Model",2:"MYCT",3:"MMIN",4:"MMAX",5:"CACH",6:"CHMIN",7:"CHMAX",8:"PRP",9:"ERP"},axis=1, inplace=True) # [,1,2,3,4,5,6,7,8,9],
import seaborn as sns

fig, axes= plt.subplots(1,1, figsize=(19,8) ) # ,xscale="log"  
sns.scatterplot(x="PRP", y="ERP", data=X0, hue="Vendor", palette="coolwarm_r"); #coolwarm spring_r tab20b
axes.legend(loc=2,)
axes.set_xscale("log")
axes.set_yscale("log")
y = X0.ERP     #(Ein-Dor)
y_mu = X0.PRP  #(Byte)
X = X0.drop(columns=["Vendor","Model" ,"ERP"] , axis=1) #
feature_names= X.columns.to_list()
X.head(2)
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y, y_mu))
print('Mean Squared Error:', metrics.mean_squared_error(y, y_mu))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y, y_mu)))
mape = np.mean(np.abs((y - y_mu) / np.abs(y_mu)))
print('Mean Absolute Percentage Error (MAPE):', round(mape * 100, 2))
print('Accuracy:', round(100*(1 - mape), 2))
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

cpu= X[["MYCT", "MMAX", "CACH", "CHMAX", "MMIN","CHMIN"]] #
cl_KM = KMeans(n_clusters=3)
y_predKM =cl_KM.fit_predict(cpu)
labels = cl_KM.labels_
cluster_centers =cl_KM.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)
fig = plt.figure(1, figsize=(11, 6))
ax = Axes3D(fig, rect=[0, 0, .999, 1], elev=28, azim=34)
#est.fit(X)  #labels = est.labels_
 
ax.scatter(X.iloc[:, 3], X.iloc[:, 1], X.iloc[:, 4],     # X[:, 2]
           c= y_predKM, edgecolor='k')     # c=labels.astype(np.float)
ax.legend(  labels); # handles=[0,1,2],=["Budget","Mid","Premium"]
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('CHMAX')
ax.set_ylabel('MMAX')
ax.set_zlabel('MMIN')
ax.set_title("CPU clusters");
#ax.dist = 8
from itertools import cycle
fig = plt.figure(1, figsize=(12, 6))
plt.clf()

colors = ["b","gold","r"]   #
for k, col in zip(range(n_clusters_), colors):
    my_members =labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X.iloc[my_members, 0], X.iloc[my_members, 1],'.', markerfacecolor=col, markersize=7.5 ) # + '.'
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.xscale('log');  plt.show()
LABS=pd.DataFrame( labels, index=X.index )
XX= X.merge(LABS, left_index=True, right_index=True)
XX.iloc[:,-1]=XX.iloc[:,-1].replace( {0:"Budget", 1:"Mid", 2:"Premium"})
sns.set_style(style="darkgrid"); fig = plt.figure(1, figsize=(12.5, 7.1))
sns.scatterplot(x="MYCT" ,y="PRP", data=XX, hue=XX.iloc[:,-1],); #,
XXerp =XX.merge( X0.ERP, left_index=True, right_index=True)
fig = plt.figure(1, figsize=(12.5, 7.1))
sns.scatterplot(x="ERP" ,y="PRP", data=XXerp, hue=XX.iloc[:,-1]); #,, hue_norm=(0, 700)
plt.xscale('log')
plt.title("3 classes by clustering"); 
cut =pd.cut(X0.PRP.values , bins= np.array([6,33,72,1200]))
XXerp["PRP_bin"] =pd.cut(X0.PRP.values , bins= np.array([6,33,72,1200]))
fig = plt.figure(1, figsize=(12.5, 7.1))
sns.scatterplot(x="ERP" ,y="PRP", data=XXerp, style=XX.iloc[:,-1],hue="PRP_bin", ); #,sizes=(2,4) ,33,72,
plt.xscale('log')
def takeloga(df_train):
    """get logarithm of dataframe, ignoring zeros """
    for col in df_train.columns:
        df_train[col] = np.where(df_train[col]>0, np.log(df_train[col]), 0)
        print(col)
    return df_train
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.22, random_state=42)
takeloga(X_train); takeloga(X_test); 
X_train.head(3)
from time import time

from sklearn.ensemble import GradientBoostingRegressor

print("Training Gradient Boosting Regressor...")
tic = time()
GBR = GradientBoostingRegressor(n_estimators=2000,max_depth=19, min_samples_leaf=4, learning_rate=0.002,random_state=1000,
                               verbose=0,max_features=0.5) #learning_rate=0.1, 
GBR.fit(X_train, y_train)
y_pred = GBR.predict(X_test) 
print("done in {:.3f}s".format(time() - tic))
print("Test R2 score: {:.2f}".format(GBR.score(X_test, y_test)))
print("Predict R2 score: {:.2f}".format(GBR.score(X_test, y_pred)))
# Use the forest's predict method on the test data
predictions = y_pred  # 
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 's.')
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
params = {'n_estimators': 2000,
          'max_depth': 4,
          'min_samples_split': 5,
          'learning_rate': 0.003,
          'loss': 'ls'}
GBR = GradientBoostingRegressor(**params)
GBR.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, GBR.predict(X_test))
print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
reg=GBR
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(X_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance  with initial log transform')
plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()
reg = GradientBoostingRegressor(**params)
reg.fit(X_train, y_train)
from sklearn.inspection import permutation_importance
reg=GBR
feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array( feature_names)[sorted_idx])
plt.title('Feature Importance (CPU)')

result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array( feature_names)[sorted_idx])  # diabetes.feature_names
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()
X_test.head(2) #
X_test.pop("PRP"); 
X_train.pop("PRP")
from sklearn.inspection import permutation_importance

feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array( feature_names)[sorted_idx])
plt.title('Feature Importance (CPU)')

result = permutation_importance(reg, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array( feature_names)[sorted_idx])  # diabetes.feature_names
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()

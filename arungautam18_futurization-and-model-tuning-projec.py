import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from scipy.stats import zscore
from sklearn.linear_model import Lasso
from sklearn.cluster import KMeans
import copy
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
conc_Data = pd.read_csv('concrete.csv')
conc_Data.shape
conc_Data.info()
conc_Data.head()
conc_Data.describe().T
# zero values present for ash ,slag and superplastic
# Looking at description of data we can see there are outliers present in data
# DIST plot to check gaissians and get idea how attributes are distributed

fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(20,20))
for i, column in enumerate(conc_Data.columns): sns.distplot(conc_Data[column],ax=axes[i//3,i%3],kde=True)
# Check outliers presence using box plot
fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(20,20))

for i, column in enumerate(conc_Data.columns): 
    sns.boxplot(conc_Data[column],ax=axes[i//3,i%3])
# outliers present for water,superplastic,slag and fineagg age

sns.pairplot(conc_Data)
# Get idea how independent attributes are releated with target variable usin corelation matrix
correlation_matrix = conc_Data.corr() 

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(correlation_matrix, annot=True, ax=ax)  

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns) 

plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns) 
# looking at coorelation matrix we cant drop any column directly,there seems to be some relation ,
# We ned to apply model and see which features are important


for i in conc_Data.columns:
    q1, q2, q3 = conc_Data[i].quantile([0.25,0.5,0.75])
    IQR = q3 - q1
    a = conc_Data[i] > q3 + 1.5*IQR
    b = conc_Data[i] < q1 - 1.5*IQR
    conc_Data[i] = np.where(a | b, q2, conc_Data[i]) 
# scaling data to apply clusetering ,lassso
conc_Data_Z = conc_Data.apply(zscore) 
# a. Identify opportunities (if any) to create a composite feature, drop a
# feature etc.
# Applying lasso regression to get idea of which attributes can be convert to approx zero coefficients 

X_scaled = conc_Data_Z.drop('strength',axis = 1)
y_scaled = conc_Data_Z['strength']
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.30, random_state=1)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train,y_train)
print ("Lasso model:", (lasso.coef_))
# lasso result: as number of attributes are less lasso model tried to make coeffiecient zero for some feature,
# but these can not be dropped directly as number of attributes are less 
# it has made zero value for coeffiecient for ash,coarseagg,fineagg
# make composit of independent attributes,and apply model with more number of attributes
# with more degree
polynomial_results = pd.DataFrame(columns={'Degree','Training data Accuracy','Testing data Accuracy'})
def get_polynomial_results(deg):
    polynomial = PolynomialFeatures(degree = deg, interaction_only=True)
    X_scaled_P = polynomial.fit_transform(X_scaled)
    X_train_P, X_test_P, y_train, y_test = train_test_split(X_scaled_P, y_scaled, test_size=0.30, random_state=1)
    
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_P,y_train)

#     regression_model = LinearRegression()
#     regression_model.fit(X_train_P, y_train)
    
    return {'Degree':deg,'Training data Accuracy':lasso.score(X_train_P, y_train),'Testing data Accuracy':lasso.score(X_test_P, y_test)}

for deg in np.arange(1,6):
    accuracy = get_polynomial_results(deg)
    polynomial_results = polynomial_results.append(accuracy,ignore_index=True)
polynomial_results
# Explore for gaussians

cluster_range = range(1,15)
cluster_errors = []
for num_clusters in cluster_range:
    clusters = KMeans(num_clusters, n_init = 5)
    clusters.fit(conc_Data_Z)
    labels = clusters.labels_
    centroids = clusters.cluster_centers_
    cluster_errors.append(clusters.inertia_)

clusters_df = pd.DataFrame({"num_clusters": cluster_range, "cluster_errors": cluster_errors})
clusters_df[0:15]
from matplotlib import cm

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
# crating 6 clusters
kmeans = KMeans(n_clusters=6, n_init = 5, random_state=12345)
kmeans.fit(conc_Data_Z)
labels = kmeans.labels_
counts = np.bincount(labels[labels>=0])
print(counts)
cluster_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))
cluster_labels['labels'] = cluster_labels['labels'].astype('category')
conc_Data_labeled = conc_Data.join(cluster_labels)

conc_Data_labeled.boxplot(by = 'labels',  layout=(3,3), figsize=(30, 20))
# feature importance
dTree = DecisionTreeRegressor(random_state=12)
dTree.fit(X_train,y_train)
print(dTree.score(X_train,y_train))
print(dTree.score(X_test,y_test))
print (pd.DataFrame(dTree.feature_importances_, columns = ["Imp"], index = X_train.columns))
conc_Data_Z_modified =  conc_Data_Z.drop(['coarseagg','ash','superplastic'],axis=1)
# with less features
X_mod_scaled = conc_Data_Z_modified.drop('strength',axis = 1)
y_mod_scaled = conc_Data_Z_modified['strength']

X_mod_train, X_mod_test, y_train, y_test = train_test_split(X_mod_scaled, y_mod_scaled, test_size=0.30, random_state=1)

dTree = DecisionTreeRegressor(random_state=12)
dTree.fit(X_mod_train,y_train)
print(dTree.score(X_mod_train,y_train))
print(dTree.score(X_mod_test,y_test))
# Still we are getting 79% of accuracy after dropping some attributes
# 4. Tuning
rf = RandomForestRegressor(random_state = 1)

rf.get_params
n_estimators = np.arange(10,100,20)
max_depth = np.arange(3,9,3)
max_features = ['auto', 'log2']
min_samples_leaf = np.arange(1,5)
bootstrap = [True, False]
# np.arange(5,50,5)
random_grid =  {'n_estimators' : n_estimators,
               'max_depth': max_depth,
               'max_features' : max_features,
                'min_samples_leaf':min_samples_leaf,
               'bootstrap' : bootstrap}
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                              n_iter = 5, scoring='neg_mean_absolute_error', 
                              cv = 3, verbose=2, random_state=42, n_jobs=-1,
                              return_train_score=True)
rf_random.fit(X_mod_train, y_train);
rf_random.best_params_
best_random = rf_random.best_estimator_
print(best_random.score(X_mod_train , y_train))
print(best_random.score(X_mod_test , y_test))
# Performance of our model has been improved by applying random search CV and also it is giving good accracy oon both training and testing data

# Applying k fold validation 
num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds, random_state=seed)
model = best_random
results = cross_val_score(model, X_mod_scaled, y_mod_scaled, cv=kfold)
print(results)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100.0, results.std()*100.0))

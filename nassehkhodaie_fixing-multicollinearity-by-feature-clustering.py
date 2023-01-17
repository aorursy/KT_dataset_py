from IPython.display import Image
import os
Image('../input/collinear-features-img/collinear_features.png')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import seaborn as sns
data = load_breast_cancer(as_frame=True)
y_data = data.target
X_data = data.data
X_data.info()
correlation_matrix = X_data.corr(method  = 'pearson')
high_cor = correlation_matrix>0.7
num_collinear_pair = (high_cor.sum().sum()-len(X_data.columns))/2
print (f'Number of collinear pairs ={num_collinear_pair}')
mask = np.array(correlation_matrix)
mask[np.tril_indices_from(mask)] = False
plt.figure(figsize=(22,22), dpi = 250)
sns.heatmap(correlation_matrix, mask  = mask,  annot=True, square=True, cmap='coolwarm')
plt.show()
# original dataset
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_data, y_data, random_state=42)
clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
clf1.fit(X_train1, y_train1)
print(f"Accuracy on test data with collinear dataset ({X_data.shape[1]} features): {clf1.score(X_test1, y_test1)}")
from collinearity_finder_treater_py import collinear_data
raw_data = collinear_data(X_data)
treated_data = raw_data.non_collinear_df(threshold = 0.7, 
                                         min_total_variance_ratio_explained=0.9, 
                                         verbose = True)
cluster_0 = raw_data.cluster_0
print (f'number of pairs in cluster_0 = {len(cluster_0.pairs)}')
print (f'number of features/nodes in cluster_0 = {len(cluster_0.nodes)}')
cluster_0.plot(fig_size=(20,20), dpi = 200, font_size = 10)
plt.show()
for cl in raw_data._clusters[1:]:
    cl.plot(fig_size = (10,6),dpi = 200, font_size = 7)
    print (cl.pairs)
# treated dataset
X_train2, X_test2, y_train2, y_test2 = train_test_split(treated_data, y_data, random_state=42)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2.fit(X_train2, y_train2)
print(f"Accuracy on test data with non-collinear dataset ({treated_data.shape[1]} features): {clf2.score(X_test2, y_test2)}")
print(f"Accuracy on test data with collinear dataset ({X_data.shape[1]} features): {clf1.score(X_test1, y_test1)}")
treated_data08 = raw_data.non_collinear_df(threshold = 0.8, 
                                         min_total_variance_ratio_explained=0.9, 
                                         verbose = False)
X_train3, X_test3, y_train3, y_test3 = train_test_split(treated_data08, y_data, random_state=42)
clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3.fit(X_train3, y_train3)
print(f"Accuracy on test data with collinear dataset ({treated_data08.shape[1]} features): {clf3.score(X_test3, y_test3)}")
raw_data.cluster_0.plot(fig_size = (20,15),dpi = 200, font_size = 15)
for cl in raw_data._clusters[1:]:
    cl.plot(fig_size = (10,5),dpi = 200, font_size = 7)
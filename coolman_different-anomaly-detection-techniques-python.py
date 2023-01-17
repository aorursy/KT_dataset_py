import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# ------ Load iris -----------
iris = pd.read_csv("../input/Iris.csv") #load the dataset
print(iris[0:2])

# ------ dataset for anomaly detection --------
X = iris.iloc[:,1:5] # ignore first column (row Id) and last column ('Species' class labels)
fig = plt.figure(figsize=(15,15))

def plot_model(lables, alg_name, plot_index):
    
    # plt.figure(plot_index)
    ax = fig.add_subplot(3,2,plot_index) 
    color_code = {'anomaly':'red', 'normal':'green'}
    colors = [color_code[x] for x in labels]

    ax.scatter(X.iloc[:,0], X.iloc[:,1], color=colors, marker='.', label='red = anomaly')
    ax.legend(loc="lower right")

    leg = plt.gca().get_legend()
    leg.legendHandles[0].set_color('red')

    ax.set_title(alg_name)

#-----------------------
# nessacry configuration for some algorithms
outliers_fraction = 0.05

# ---------------- DBSCAN -----------
from sklearn.cluster import DBSCAN

# eps: maximum distance between two samples
model = DBSCAN(eps=0.63).fit(X)
labels = model.labels_
# print(labels)
# label == -1 then it is anomaly
labels = [('anomaly' if x==-1 else 'normal') for x in labels]
# print(labels)
plot_model(labels, 'DBSCAN', 1)

#----------- Isolation Forest --------------------------
from sklearn.ensemble import IsolationForest
from scipy import stats

model = IsolationForest().fit(X)
scores_pred = model.decision_function(X)
threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)

labels = [('anomaly' if x<threshold else 'normal') for x in scores_pred]
plot_model(labels, 'Isolation Forest', 2)

# ----------  LocalOutlierFactor -----------
from sklearn.neighbors import LocalOutlierFactor 

model = LocalOutlierFactor()
model.fit_predict(X)
scores_pred = model.negative_outlier_factor_
threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)

labels = [('anomaly' if x<threshold else 'normal') for x in scores_pred]
plot_model(labels, 'LocalOutlierFactor', 3)
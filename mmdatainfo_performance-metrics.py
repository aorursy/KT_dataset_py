# Classification

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score



# Regression

from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

from sklearn.metrics import explained_variance_score, r2_score 



# Clustering

from sklearn.metrics import adjusted_rand_score, homogeneity_score, adjusted_mutual_info_score



# other libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# Use vector drawing inside jupyter notebook

%config InlineBackend.figure_format = "svg"

# Set matplotlib default axis font size (inside this notebook)

plt.rcParams.update({'font.size': 8})
fpr, tpr, threshold = roc_curve([1  ,1  ,1  ,1  ,1  ,0  ,0  ,0  ,0  ,0],

                                [0.4,0.5,0.6,0.7,0.5,0.3,0.3,0.3,0.2,0.1])
print("FPR       = ",fpr)

print("TPR       = ",tpr)

print("Threshold = ",threshold)

# Add area under curve score. Could use `roc_auc_score` directly without `fpr,tdr`

print("AUC       = ",auc(fpr,tpr))
plt.figure(figsize=(2.5,2.5))

plt.plot(fpr,tpr,"k--");

# Color-code the markers with maximum at "true" threshold (yellow)

plt.scatter(fpr,tpr,c=-abs(threshold-0.4))

plt.axis("equal");

plt.title("Prefect ROC");
y_true = [2,2,2,2,1,1,1,1,1,0,0,0];

y_pred = [2,1,0,1,1,1,0,1,1,1,1,0];

for (n,s) in zip(["Accuracy score","Balanced accuracy","Classification report"],

                 [accuracy_score,balanced_accuracy_score,classification_report]):

    print(n," : \n",s(y_true,y_pred))
def rmse(y1,y2):

    return np.sqrt(mean_squared_error(y1,y2))



def std(y1,y2):

    return np.std(y1-y2)
def evalregmetric(outlier=True,subt_mean=True):

    # create a copy of prediction and modify it according to settings

    y_use = y_pred.copy();

    if outlier == True:

        y_use[-1] = y_use[-1]+4;

    if subt_mean == True:

        y_use = y_use + np.mean(y_true-y_use);

    # run loop over all metrics

    out = np.array([]);

    for metric in [mean_squared_error,rmse,std,mean_absolute_error,median_absolute_error,

                         explained_variance_score,r2_score]:

        out = np.append(out,metric(y_true,y_use));

    return out
# Declare true and predicted values

y_true = np.array([2,2,2,2,1,1,1,1,1,0,0,0]);

y_pred = np.array([2,1,0,1,1,1,0,1,1,1,1,0]);



# Store the result in a dataframe for better visualisation (table)

result = pd.DataFrame(np.array(["MSE","RMSE","STD","MAE","MedAE","EV","R2"]),columns=["Metric"])

result = pd.DataFrame(result).set_index("Metric")



# Run for all combinations: Outlier/Mean subtraction

for (o,m) in zip([False,True,False,True],[False,False,True,True]):

    temp = pd.DataFrame({"Metric": np.array(["MSE","RMSE","STD","MAE","MedAE","EV","R2"]),

     "outlier:"+str(o)+" mean:"+str(m): evalregmetric(outlier=o,subt_mean=m)})

    result = result.merge(pd.DataFrame(temp).set_index("Metric"),left_index=True,right_index=True)

# Show result

result.round(3)
# Synthetic clustiring data with 3 clusters (1,2,3 labels)

a = np.array([1,2,1,1,1,2,3,3,2,2,1,2]);

b = np.array([1,1,2,1,1,3,3,3,3,2,1,2]);

for (n,m) in zip(["ARI","AMI","Homogeneity"],

                 [adjusted_rand_score,adjusted_mutual_info_score,homogeneity_score]):

    print(n," = {}".format(np.round(m(a,b),3)))
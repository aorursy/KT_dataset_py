import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
# Viewing the columns
print(data.columns)
print(data.shape)
print(data.describe())
# Ploting the histogram of each parameter
"""
A histogram is a graphical display of data using bars of different heights. 
In a histogram, each bar groups numbers into ranges. 
Taller bars show that more data falls in that range. 
A histogram displays the shape and spread of continuous sample data.
"""
data.hist(figsize=(20,20))
plt.show()
# viewing the amound of fraudulent cases
data['Class'].value_counts()[0]
# Outlier fraction
fraud = data['Class'].value_counts()[1]
genuine = data['Class'].value_counts()[0]
outlier_frac = fraud/len(data)
print(outlier_frac)
# Correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat,vmax=.8,square=True)
plt.show()
# Seperating the labels from the data
data_copy = data.copy(deep=True)
Y = data['Class']
data.drop('Class',axis = 1,inplace = True)
X = data
print(X.shape)
print(Y.shape)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# define a random state
state =1

#define the outlier detection methods
classifiers = {
    "Isolation Forest" : IsolationForest(
                            max_samples=len(X),
                            contamination = outlier_frac,
                            random_state = state
                        ),
    "Local Outlier Factor" : LocalOutlierFactor(
                                n_neighbors = 20,
                                contamination = outlier_frac,
                            )
}
# Fit the model 
n_outlier = fraud

for i, (clf_name,clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # Reshape the prediction values to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    # classification metrics
    print(f'{clf_name} : {n_errors}')
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y,y_pred))
import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Seaborn: {}'.format(seaborn.__version__))
print('Scipy: {}'.format(scipy.__version__))
print('Sklearn: {}'.format(sklearn.__version__))
#importing necesssry packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import sklearn as sk

#load the data from csv file 
data=pd.read_csv("../input/creditcard.csv")
data.head()
#exploring data
print(data.columns)
print(data.shape)

#Time: Time taken in completing transaction
#Amount: Amount of transaction
#Class: 0 indicates a valid transaction while 1 indicates an invalid transaction
#Other columns are properties extracted from transactions
print(data.describe())
data=data.sample(frac=0.1, random_state=1)
print(data.shape)
#plot the histogram for each parameter
data.hist(figsize=(20,20))
plt.show()
#determine the number of fraud cases in dataset
Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]

outlier_fraction = len(Fraud)/(float)(len(Valid))
print(outlier_fraction)

print('Fraud cases: {}'.format(len(Fraud)))
print('Valid cases: {}'.format(len(Valid)))

#correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12,9))
sns.heatmap(corrmat, vmax =.8, square = True)
plt.show()
#Fiter the column to remove the data we do not want
columns = data.columns.tolist()
columns = [c for c in columns if c not in ['Class']]

target = 'Class'

X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

#define a random state
state = 1

#define the outlier detection method
classifiers = {
    "Isolation Forest" : IsolationForest(max_samples = len(X),
                                       contamination = outlier_fraction,
                                       random_state = state),
    "Local Outlier Factor" : LocalOutlierFactor(n_neighbors = 20,
                                               contamination = outlier_fraction)
}

#fitting the model
n_outliers = len(Fraud)


for i, (clf_name, clf) in enumerate(classifiers.items()):
    #fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else: 
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    #resetting the predicted value to 0 for valid, 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    #run classification matrix
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


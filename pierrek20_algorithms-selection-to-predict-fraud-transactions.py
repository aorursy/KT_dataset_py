# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from time import time

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from scipy.spatial.distance import cdist

# Algorithmes 
from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

#Common Model Helpers
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics 
#---------------------- Import data
data = pd.read_csv('../input/creditcard.csv')
pd.set_option('display.max_columns', None)
print(data.head())
print(50*'-')
print(data.shape)
number_fraud_transaction = len(data[data.Class == 1])
number_normal_transaction = len(data[data.Class == 0])

print('number of fraud transactions: ',number_fraud_transaction)
print(20*'-')
print('number of normal transactions: ',number_normal_transaction)
# get the index of the normal transactions : 
normal_transaction_index = data[data.Class == 0].index
normal_transaction_index = np.array(normal_transaction_index)
# get the index of the fraud transactions : 
fraud_transaction_index = data[data.Class == 1].index
fraud_transaction_index = np.array(fraud_transaction_index)
random_normal_transaction_index = np.random.choice(normal_transaction_index,number_fraud_transaction, replace = False)
random_normal_transaction_index = np.array(random_normal_transaction_index)
selection_index = np.concatenate([fraud_transaction_index,random_normal_transaction_index])
selection_data = data.iloc[selection_index,:]
print( pd.DataFrame ( {'NB' : selection_data.groupby(['Class']).size()}).reset_index()) 
X = selection_data.drop(['Class'], axis = 1)
y = selection_data[['Class']]

X = np.array(X)
y = np.array(y)
# Import algorithms 
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),  
    ]
# Dataframe to compare algorithms
col = ['MLA Name','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Time']
MLA_compare = pd.DataFrame(columns = col)

# Cross validation split
cv_split = model_selection.ShuffleSplit(n_splits = 5, test_size = .2, train_size = .8, random_state = 0 )


index = 0 
for alg in MLA: 
    # Nom de l'algo
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[index, 'MLA Name'] = MLA_name
    
    # Cross validation
    cv_results = model_selection.cross_validate(alg, X, y, cv  = cv_split)
    MLA_compare.loc[index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()
    MLA_compare.loc[index,'MLA Time'] = cv_results['fit_time'].mean()

    index +=1
# Print comparison table
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
print(MLA_compare)

sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'blue')
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')

plt.show()
best_score = max(MLA_compare['MLA Test Accuracy Mean'])

best_MLA = MLA_compare[MLA_compare['MLA Test Accuracy Mean'] == best_score].reset_index(drop = True)

name_best_alg = np.array(best_MLA['MLA Name'])
name_best_alg = name_best_alg[0]
print(name_best_alg)
for alg in MLA : 
    name = alg.__class__.__name__
    if name == name_best_alg:
        alg.fit(X,y)
        #prediction = alg.predict(X_prediction)
        #print(prediction)



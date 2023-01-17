import matplotlib.pyplot as plt 

import numpy as np 

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import scale

import xgboost as xgb

## Tunin parameters

# Grid Search for tuning parameters

from sklearn.model_selection import GridSearchCV

# RandomizedSearch for tuning (possibly faster than GridSearch)

from sklearn.model_selection import RandomizedSearchCV

# Bayessian optimization supposedly faster than GridSearch

from bayes_opt import BayesianOptimization

import scipy

# Confusion matrix for model assessment

from sklearn.metrics import confusion_matrix

# accuracy_score to calculate the... accuracy score

from sklearn.metrics import accuracy_score



## Benchmarkers

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier



## Because somewhat ~normally~ distributed I think?

from sklearn.naive_bayes import GaussianNB

# Non-parametric

from sklearn.tree import DecisionTreeClassifier



## Ensembles

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



## NeuralNet 

from sklearn.neural_network import MLPClassifier

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/data_set_ALL_AML_train.csv')

test = pd.read_csv('../input/data_set_ALL_AML_independent.csv')

patient_cancer = pd.read_csv('../input/actual.csv')
test.head()
# Remove "call" columns from training a test dataframes

train_keepers = [col for col in train.columns if "call" not in col]

test_keepers = [col for col in test.columns if "call" not in col]



train = train[train_keepers]

test = test[test_keepers]
# Transpose the columns and rows so that genes become features and rows become observations

train = train.T

test = test.T

train.head()
# Clean up the column names for training data

train.columns = train.iloc[1]

train = train.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)



# Clean up the column names for training data

test.columns = test.iloc[1]

test = test.drop(["Gene Description", "Gene Accession Number"]).apply(pd.to_numeric)



train.head()
# Reset the index. The indexes of two dataframes need to be the same before you combine them

train = train.reset_index(drop=True)



# Subset the first 38 patient's cancer types

pc_train = patient_cancer[patient_cancer.patient <= 38].reset_index(drop=True)



# Combine dataframes for first 38 patients: Patient number + cancer type + gene expression values

train = pd.concat([pc_train,train], axis=1)





# Handle the test data for patients 38 through 72

# Clean up the index

test = test.reset_index(drop=True)



# Subset the last patient's cancer types to test

pc_test = patient_cancer[patient_cancer.patient > 38].reset_index(drop=True)



# Combine dataframes for last patients: Patient number + cancer type + gene expression values

test = pd.concat([pc_test,test], axis=1)
test.head()
sample = train.iloc[:,2:].sample(n=100, axis=1)

sample["cancer"] = train.cancer

sample.describe().round()

sample = sample.drop("cancer", axis=1)

sample.plot(kind="hist", legend=None, bins=20, color='k')

sample.plot(kind="kde", legend=None)
scaler = StandardScaler().fit(train.iloc[:,2:])

scaled_train = scaler.transform(train.iloc[:,2:])

scaled_test = scaler.transform(test.iloc[:,2:])



x_train = train.iloc[:,2:]

y_train = train.iloc[:,1]

x_test = test.iloc[:,2:]

y_test = test.iloc[:,1]
sample_scaled = pd.DataFrame((sample))

sample_scaled.plot(kind="hist", normed=True, legend=None, bins=10, color='k')

sample_scaled.plot(kind="kde", legend=None)
# Logistic Regression

# Paramaters

lr_param = {

    "C":[0.01, 0.1, 10, 100],

    "fit_intercept":[True, False],

    "warm_start":[True,False],

    "random_state":[1]

} 



# SVM

dsvm_param = {

    "C": scipy.stats.expon(scale=.01),

    "gamma": scipy.stats.expon(scale=.01),

    "kernel": ["rbf"],

    "random_state": [1]

}



dknn_param = {

    "n_neighbors": scipy.stats.randint(1,33),

    "weights": ["uniform", "distance"],

    "algorithm": ["ball_tree", "kd_tree", "brute"],

    "leaf_size": scipy.stats.randint(1,1000),

    "p": [1,2]

}









dlr_param = {

    "C":scipy.stats.expon(scale=.01),

    "fit_intercept":[True, False],

    "warm_start":[True,False],

    "random_state":[1]

} 



# SVM

svm_param = {

    "C": [.01, .1, 1, 5, 10, 100],

    "gamma": [0, .01, .1, 1, 5, 10, 100],

    "kernel": ["rbf"],

    "random_state": [1]

}



knn_param = {

    "n_neighbors": [i for i in range(1,30,5)],

    "weights": ["uniform", "distance"],

    "algorithm": ["ball_tree", "kd_tree", "brute"],

    "leaf_size": [1, 10, 30],

    "p": [1,2,3]

}



dtc_param = {

    "max_depth": [None],

    "min_samples_split": [2],

    "min_samples_leaf": [1],

    "min_weight_fraction_leaf": [0.],

    "max_features": [None],

    "random_state": [4],

    "max_leaf_nodes": [None], # None = infinity or int

    "presort": [True, False]

}

rf_param = {

    "n_estimators": [10,500,1000],

    "criterion": ["gini","entropy"],

    "max_features": ["auto"],

    "max_depth": [None,1,5,10],

    "max_leaf_nodes": [None],

    "oob_score": [False],

    "n_jobs": [-1],

    "warm_start": [False]

}

xgb_param = {'nthread':[-1], #when use hyperthread, xgboost may become slower

              'subsample': [.1, .4, 0.8],

              'n_estimators': [100,1000], #number of trees, change it to 1000 for better results

            }
rf = RandomForestClassifier()

dtc = DecisionTreeClassifier()

knn = KNeighborsClassifier()

svm = SVC()

lr = LogisticRegression()

xgb = xgb.XGBClassifier()

# Not testing everything because otherwise it times out or my wifi dies (Virgin Medias Uptime is trash)

classy =[[rf, rf_param], [svm, svm_param], [xgb,xgb_param]]

dclassy =[[knn, dknn_param], [lr, dlr_param], [svm, dsvm_param]]


def grid_search(x, params=classy):

    opt = 0

    best_score = 0

    for i in params:

        clf = GridSearchCV(estimator = i[0],param_grid = i[1], cv=10, n_jobs=-1, verbose=1, return_train_score=True)

        clf.fit(x, y_train)

        if clf.best_score_ > best_score:

            opt = clf

    return opt



def random_search(x, params=dclassy):

    opt = 0

    best_score = 0

    for i in params:



        clf = RandomizedSearchCV(estimator = i[0],param_distributions = i[1], n_iter=10,n_jobs=-1,cv=10,verbose=1,random_state=1,return_train_score=True)

        clf.fit(x, y_train)     

        if clf.best_score_ > best_score:

            opt = clf

    return opt

classy = 0

fscore = 0

def full_search(d=[[scaled_train, scaled_test, "Scaled"]]):

    for data in d:

        grid = grid_search(x=data[0])

        random = random_search(x=data[0])

        grid_score = accuracy_score(y_test, grid.predict(data[1])) 

        random_score = accuracy_score(y_test, random.predict(data[1]))

        if grid_score > random_score:

            if grid_score > fscore:

                fscore = grid_score

                classy = grid 

        else:

            if random_score > fscore:

                classy = random

                fscore = random_score

    for data in d:

        print("Data modifications: ",data[2])

        print("Accuracy Score: ", accuracy_score(y_test ,classy.predict(data[1])))



    
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

PCA_train = pca.fit(scaled_train)

PCA_test = pca.fit(scaled_test)



tsne_train = TSNE(n_components=2).fit_transform(scaled_train)

tsne_test = TSNE(n_components=2).fit_transform(scaled_test)

#Since we are using the same test we dont have to actually worry about passing the test parameters

#For good practice you'd normally pass additional the answer key as well 

dim_reduced_data = [[PCA_train, PCA_test, "PCA"], [tsne_train, tsne_test, "TSNE"]]

full_search(d=dim_reduced_data)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print("Two datasets:")

print(check_output(["ls", "../input/"]).decode("utf8"))



print("1)")

print(check_output(["ls", "../input/additional-info-leukemia/"]).decode("utf8"))



print("2) [original data]")

print(check_output(["ls", "../input/gene-expression/"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# importing train dataset

raw_dataset_train= pd.read_csv('../input/gene-expression/data_set_ALL_AML_train.csv')



#importing test dataset

raw_dataset_test= pd.read_csv('../input/gene-expression/data_set_ALL_AML_independent.csv')



#importing cancer types

cancer_types = pd.read_csv('../input/gene-expression/actual.csv')
# removing the 'call' tags

dataset_tags_train = [col for col in raw_dataset_train if "call" not in col]

dataset_tags_test = [col for col in raw_dataset_test if "call" not in col]



# put patients in rows and gene expression by gene in columns

train_dataset = raw_dataset_train[dataset_tags_train].set_index("Gene Accession Number").T

test_dataset = raw_dataset_test[dataset_tags_test].set_index("Gene Accession Number").T





# removing chip endogenous controls (not informative for cancer classification)

import re

dataset_tags_train = [col for col in train_dataset if not re.match("^AFFX", col)]

dataset_tags_test = [col for col in test_dataset if not re.match("^AFFX", col)]



train_dataset = train_dataset[dataset_tags_train]

test_dataset = test_dataset[dataset_tags_test]



# clean the column names

train_dataset = train_dataset.drop(["Gene Description"])

test_dataset = test_dataset.drop(["Gene Description"])



#test_dataset.head()

train_dataset.head()

#raw_dataset_train.head()
# Reset the index. The indexes of two dataframes need to be the same before you combine them

train_dataset = train_dataset.reset_index(drop=True)



# Subset the first 38 patient's cancer types

ct_train = cancer_types[cancer_types.patient <= 38].reset_index(drop=True)



# Combine dataframes for first 38 patients: Patient number + cancer type + gene expression values

train_dataset = pd.concat([ct_train,train_dataset], axis=1)





# Handle the test data for patients 38 through 72

# Clean up the index

test_dataset = test_dataset.reset_index(drop=True)



# Subset the last patient's cancer types to test

ct_test = cancer_types[cancer_types.patient > 38].reset_index(drop=True)



# Combine dataframes for last patients: Patient number + cancer type + gene expression values

test_dataset = pd.concat([ct_test,test_dataset], axis=1)



train_dataset.head()
x_train = train_dataset.iloc[:,2:]

y_train = train_dataset.iloc[:,1]

x_test = test_dataset.iloc[:,2:]

y_test = test_dataset.iloc[:,1]
print(len(y_test), "patients in test set.", len(x_test.columns), "expression values per patient.")

print(len(y_train), "patients in train set.", len(x_train.columns), "expression values per patient.")
# scaling features

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler().fit(train_dataset.iloc[:,2:])

scaled_train = scaler.transform(train_dataset.iloc[:,2:])

scaled_test = scaler.transform(test_dataset.iloc[:,2:])

len_test = len(y_test)

prob_random = 1/2 ** len_test

print("Test set contains", len_test, "patients.")

print("The probability of predict correctly the cancer type of all patients is:", prob_random )
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC # "Support vector classifier"



svm_param = {

    "C": [.01, .1, 1, 5, 10, 100],

    "gamma": [0, .01, .1, 1, 5, 10, 100],

    "kernel": ["rbf"],

    "random_state": [1]

}



svm_model = GridSearchCV(estimator=SVC(), 

                               param_grid=svm_param, 

                               scoring=None,

                               n_jobs=-1, 

                               cv=10, 

                               verbose=1,

                               return_train_score=True)



svm_model.fit(scaled_train, y_train)



print("Best score with GridSearchCV:", svm_model.best_score_)

print(svm_model.best_estimator_, "\n")
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score



y_pred = svm_model.predict(scaled_test)

#y_test = y_test.tolist()

accu_svm = accuracy_score(y_test, y_pred)

prec_svm_all = precision_score(y_test, y_pred, pos_label="ALL")

prec_svm_aml = precision_score(y_test, y_pred, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy SVM (rbf):",accu_svm)

print("Precision SVM (rbf) ALL:",prec_svm_all)

print("Precision SVM (rbf) AML:",prec_svm_aml)



#cross validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut

scores = cross_val_score(svm_model, scaled_test, y_test, cv=10)

score_mean = scores.mean()



print("Accuracy SVM (cross-val): ",score_mean)
# Naive Bayes Gaussian

from sklearn.naive_bayes import GaussianNB

model_naivebayes = GaussianNB()

model_naivebayes.fit(scaled_train, y_train)

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score



y_pred_bayes = model_naivebayes.predict(scaled_test)

accu_nb = accuracy_score(y_test, y_pred_bayes, normalize=True)

prec_nb_all = precision_score(y_test, y_pred_bayes, pos_label="ALL")

prec_nb_aml = precision_score(y_test, y_pred_bayes, pos_label="AML")



#print(y_pred)

#print(y_test)



print("Accuracy NB:",accu_nb)

print("Precision NB ALL:",prec_nb_all)

print("Precision NB AML:",prec_nb_aml)
# support vector machine grid search

from sklearn.svm import SVC

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline



pca = PCA(svd_solver='randomized', n_components=300, whiten=True, random_state=1)

svc = SVC(kernel='rbf', class_weight='balanced')

model_svm_pca = make_pipeline(pca, svc)

#model_svm_pca = svc
from sklearn.model_selection import GridSearchCV

param_grid = {'svc__C': [0.005, 0.01, 0.5,1, 2.5, 5, 7.5, 10, 50],

              'svc__gamma': [0.000005, 0.00005, 0.0001, 0.0005, 0.001, 0.005]}

grid = GridSearchCV(model_svm_pca, param_grid)



%time grid.fit(scaled_train, y_train)

print(grid.best_params_)
model_svm_pca = grid.best_estimator_

y_pred_svm_pca = model_svm_pca.predict(scaled_test)

accu_svm_pca = accuracy_score(y_test, y_pred_svm_pca)

prec_svm_pca_all = precision_score(y_test, y_pred_svm_pca, pos_label="ALL")

prec_svm_pca_aml = precision_score(y_test, y_pred_svm_pca, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy PCA-SVM (rbf):",accu_svm_pca)

print("Precision PCA-SVM (rbf) ALL:",prec_svm_pca_all)

print("Precision PCA-SVM (rbf) AML:",prec_svm_pca_aml)
# Decision tree classifier

from sklearn.tree import DecisionTreeClassifier

dtc_model = DecisionTreeClassifier()



dtc_param = {

    "max_depth": [None],

    "min_samples_split": [2],

    "min_samples_leaf": [1],

    "min_weight_fraction_leaf": [0.],

    "max_features": [None],

    "random_state": [1],

    "max_leaf_nodes": [None], # None = infinity or int

    "presort": [True, False]

}



dtc = GridSearchCV(estimator=dtc_model, param_grid=dtc_param, 

                               scoring=None,

                               n_jobs=-1, 

                               cv=10, 

                               verbose=1,

                               return_train_score=True)





%time dtc.fit(scaled_train, y_train)

print(dtc.best_params_)

#print(dtc.best_score_)

print(dtc.best_estimator_)
dtc_model = dtc.best_estimator_

y_pred_dtc = dtc_model.predict(scaled_test)

accu_dtc = accuracy_score(y_test, y_pred_dtc)

prec_dtc_all = precision_score(y_test, y_pred_svm_pca, pos_label="ALL")

prec_dtc_aml = precision_score(y_test, y_pred_svm_pca, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy Decision Tree:",accu_dtc)

print("Precision Decision Tree ALL:",prec_dtc_all)

print("Precision Decision Tree AML:",prec_dtc_aml)
# random forest

from sklearn.ensemble import RandomForestClassifier



rf_param = {

    "n_estimators": [1,10,50,100,500,1000],

    "criterion": ["gini","entropy"],

    "max_features": ["auto"],

    "max_depth": [None,1,5,10],

    "max_leaf_nodes": [None],

    "oob_score": [False],

    "n_jobs": [-1],

    "warm_start": [False],

    "random_state": [1]

}



rf_model = RandomForestClassifier()



rf = GridSearchCV(estimator=rf_model, param_grid=rf_param, 

                               scoring=None,

                               n_jobs=-1, 

                               cv=10, 

                               verbose=1,

                               return_train_score=True)





%time rf.fit(scaled_train, y_train)



print("Best score:", rf.best_score_)

print(rf.best_estimator_)
rf_model = rf.best_estimator_

y_pred_rf = rf_model.predict(scaled_test)

accu_dtc = accuracy_score(y_test, y_pred_rf, normalize=True)

prec_dtc_all = precision_score(y_test, y_pred_rf, pos_label="ALL")

prec_dtc_aml = precision_score(y_test, y_pred_rf, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy Random Forest:",accu_dtc)

print("Precision Random Forest ALL:",prec_dtc_all)

print("Precision Random Forest AML:",prec_dtc_aml)
from sklearn.neighbors import KNeighborsClassifier



knn_param = {

    "n_neighbors": [i for i in range(1,30,5)],

    "weights": ["uniform", "distance"],

    "algorithm": ["ball_tree", "kd_tree", "brute"],

    "leaf_size": [1, 10, 30],

    "p": [1,2]

}



knn_model = KNeighborsClassifier()



knn_grid = GridSearchCV(estimator=knn_model, param_grid=knn_param, 

                               scoring=None,

                               n_jobs=-1, 

                               cv=10, 

                               verbose=1,

                               return_train_score=True)



%time knn_grid.fit(scaled_train, y_train)



print("Best score:", knn_grid.best_score_)

print(knn_grid.best_estimator_, "\n")



knn_model = knn_grid.best_estimator_

y_pred_knn = knn_model.predict(scaled_test)

accu_knn = accuracy_score(y_test, y_pred_knn, normalize=True)

prec_knn_all = precision_score(y_test, y_pred_knn, pos_label="ALL")

prec_knn_aml = precision_score(y_test, y_pred_knn, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy KNN:",accu_knn)

print("Precision KNN ALL:",prec_knn_all)

print("Precision KNN AML:",prec_knn_aml)
#cross validation

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import LeaveOneOut

scores = cross_val_score(knn_model, scaled_test, y_test, cv=10)

score_mean = scores.mean()



print(score_mean)

gnames = open('../input/additional-info-leukemia/mart_gene_equivalencies.txt')



gnames_dict = {}

for line in gnames:

    cur_line = line.strip().split("\t")

    if len(cur_line) >= 5:

        #print(cur_line)

        symbol = cur_line[3]

        micro_id = cur_line[4]

        if symbol != "" and micro_id != "":

            if gnames_dict.get(symbol) != None:

                if len(gnames_dict.get(symbol)) != 0 and micro_id not in gnames_dict.get(symbol):

                    gnames_dict[symbol].append(micro_id)

            else:

                gnames_dict[symbol] = [micro_id]



print("Total of",len(gnames_dict), "genes asociated to at least one microarray ID of Affymetrix chip")
aml_datafile = '../input/additional-info-leukemia/AML_gene_tags.txt'



aml_gene_set = set()

with open(aml_datafile) as aml_data:

    for line in aml_data:

        gene = line.strip().split(";")[0]

        aml_gene_set.add(gene)

        

print(aml_gene_set)

print("Total of", len(aml_gene_set), "genes in AML pathway from KEGG database.")
aml_affyIDs = {}

for aml_gene in aml_gene_set:

    if gnames_dict.get(aml_gene) != None:

        aml_affyIDs[aml_gene] = gnames_dict.get(aml_gene)

    #affy_id = gnames.loc[aml_gene, "HGNC symbol"]

    

#print(aml_affyIDs)

aml_microIDs = []

for micro_list in aml_affyIDs.values():

    for micro_id in micro_list:

        aml_microIDs.append(micro_id)

        

#print(aml_microIDs)

print("Those genes are associated to",len(aml_microIDs),"microarray IDs from the experiment.")
# removing the 'call' tags

reduced_tags_train = [col for col in raw_dataset_train if "call" not in col]

reduced_tags_test = [col for col in raw_dataset_test if "call" not in col]



# put patients in rows and gene expression by gene in columns

train_reduced = raw_dataset_train[reduced_tags_train].set_index("Gene Accession Number").T

test_reduced = raw_dataset_test[reduced_tags_test].set_index("Gene Accession Number").T





# removing chip endogenous controls (not informative for cancer classification)

import re

reduced_tags_train = [col for col in train_reduced if col in aml_microIDs]

reduced_tags_test = [col for col in test_reduced if col in aml_microIDs]



train_reduced = train_reduced[reduced_tags_train]

test_reduced = test_reduced[reduced_tags_test]



# clean the column names

train_reduced = train_reduced.drop(["Gene Description"])

test_reduced = test_reduced.drop(["Gene Description"])



# Reset the index. The indexes of two dataframes need to be the same before you combine them

train_reduced = train_reduced.reset_index(drop=True)



# Subset the first 38 patient's cancer types

ct_train = cancer_types[cancer_types.patient <= 38].reset_index(drop=True)



# Combine dataframes for first 38 patients: Patient number + cancer type + gene expression values

train_reduced = pd.concat([ct_train,train_reduced], axis=1)





# Handle the test data for patients 38 through 72

# Clean up the index

test_reduced = test_reduced.reset_index(drop=True)



# Subset the last patient's cancer types to test

ct_test = cancer_types[cancer_types.patient > 38].reset_index(drop=True)



# Combine dataframes for last patients: Patient number + cancer type + gene expression values

test_reduced = pd.concat([ct_test,test_reduced], axis=1)



train_reduced.head()

x_train_r = train_reduced.iloc[:,2:]

y_train_r = train_reduced.iloc[:,1]

x_test_r = test_reduced.iloc[:,2:]

y_test_r = test_reduced.iloc[:,1]
print(len(y_test_r), "patients in test set.", len(x_test_r.columns), "expression values per patient.")

print(len(y_train_r), "patients in train set.", len(x_train_r.columns), "expression values per patient.")
# scaling features

from sklearn.preprocessing import StandardScaler



scaler_r = StandardScaler().fit(train_reduced.iloc[:,2:])

scaled_train_r = scaler_r.transform(train_reduced.iloc[:,2:])

scaled_test_r = scaler_r.transform(test_reduced.iloc[:,2:])
# random forest

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier



rf_param = {

    "n_estimators": [1,10,50,100,500,1000],

    "criterion": ["gini","entropy"],

    "max_features": ["auto"],

    "max_depth": [None,1,5,10],

    "max_leaf_nodes": [None],

    "oob_score": [False],

    "n_jobs": [-1],

    "warm_start": [False],

    "random_state": [1]

}



rf_model_r = RandomForestClassifier()



rf_r = GridSearchCV(estimator=rf_model_r, param_grid=rf_param, 

                               scoring=None,

                               n_jobs=-1, 

                               cv=10, 

                               verbose=1,

                               return_train_score=True)





%time rf_r.fit(scaled_train_r, y_train_r)



print("Best score:", rf_r.best_score_)

print(rf_r.best_estimator_)
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score



rf_model_r = rf_r.best_estimator_

y_pred_r_rf = rf_model_r.predict(scaled_test_r)

accu_rf_r = accuracy_score(y_test_r, y_pred_r_rf, normalize=True)

prec_rf_all_r = precision_score(y_test_r, y_pred_r_rf, pos_label="ALL")

prec_rf_aml_r = precision_score(y_test_r, y_pred_r_rf, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy Random Forest for reduced dataset:",accu_rf_r)

print("Precision Random Forest ALL (rd):",prec_rf_all_r)

print("Precision Random Forest AML (rd):",prec_rf_aml_r)
# Decision tree classifier

from sklearn.tree import DecisionTreeClassifier

dtc_model_r = DecisionTreeClassifier()



dtc_param = {

    "max_depth": [None],

    "min_samples_split": [2],

    "min_samples_leaf": [1],

    "min_weight_fraction_leaf": [0.],

    "max_features": [None],

    "random_state": [1],

    "max_leaf_nodes": [None], # None = infinity or int

    "presort": [True, False]

}



dtc_r = GridSearchCV(estimator=dtc_model_r, param_grid=dtc_param, 

                               scoring=None,

                               n_jobs=-1, 

                               cv=10, 

                               verbose=1,

                               return_train_score=True)





%time dtc_r.fit(scaled_train_r, y_train_r)

print(dtc_r.best_params_)

#print(dtc.best_score_)

print(dtc_r.best_estimator_)
dtc_model_r = dtc_r.best_estimator_

y_pred_dtc_r = dtc_model_r.predict(scaled_test_r)

accu_dtc_r = accuracy_score(y_test_r, y_pred_dtc_r)

prec_dtc_all_r = precision_score(y_test_r, y_pred_dtc_r, pos_label="ALL")

prec_dtc_aml_r = precision_score(y_test_r, y_pred_dtc_r, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy Decision Tree reduced dataset:",accu_dtc_r)

print("Precision Decision Tree ALL (rd):",prec_dtc_all_r)

print("Precision Decision Tree AML (rd):",prec_dtc_aml_r)
from sklearn.svm import SVC # "Support vector classifier"



svm_param = {

    "C": [.01, .1, 1, 5, 10, 100],

    "gamma": [0, .01, .1, 1, 5, 10, 100],

    "kernel": ["rbf"],

    "random_state": [1]

}



svm_model_r = GridSearchCV(estimator=SVC(), 

                               param_grid=svm_param, 

                               scoring=None,

                               n_jobs=-1, 

                               cv=10, 

                               verbose=1,

                               return_train_score=True)



svm_model_r.fit(scaled_train_r, y_train_r)



print("Best score with GridSearchCV:", svm_model_r.best_score_)

print(svm_model_r.best_estimator_, "\n")
y_pred_r = svm_model_r.predict(scaled_test_r)



accu_svm_r = accuracy_score(y_test_r, y_pred_r)

prec_svm_all_r = precision_score(y_test_r, y_pred_r, pos_label="ALL")

prec_svm_aml_r = precision_score(y_test_r, y_pred_r, pos_label="AML")



#print(y_pred)

#print(y_test)

print("Accuracy SVM (rbf) reduced dataset:",accu_svm_r)

print("Precision SVM (rbf) ALL (rd):",prec_svm_all_r)

print("Precision SVM (rbf) AML (rd):",prec_svm_aml_r)



#cross validation

#from sklearn.model_selection import cross_val_score

#from sklearn.model_selection import LeaveOneOut

#scores = cross_val_score(svm_model, scaled_test, y_test, cv=10)

#score_mean = scores.mean()



#print("Accuracy SVM (cross-val): ",score_mean)
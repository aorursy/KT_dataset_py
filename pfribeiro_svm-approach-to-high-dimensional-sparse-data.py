# The libraries needed in the project

import pandas as pd

from sklearn.svm import LinearSVC

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report

from statistics import mean

from sklearn.model_selection import StratifiedKFold

from sklearn import preprocessing



data = pd.read_csv("../input/covid19-csv/dataset_csv.CSV", encoding ='latin1')



# Initial contact with the data

data.dtypes

data.info()

data.describe()
nonnumeric_data = data.select_dtypes(include='object')



# Run, for each column, and gather the string values, and their appropriate numeric substitutes.

# There are many options available, the ones used in here are only 2.



# Either the value is probabilistic treated (ie. probability of testing positive)

# or the feature is simple, and it can be mapped directly to any number (synthetic category)

# given the fact that those columns can easily be normalized and standardized



# Analysis of common values for all non-numeric features present in the data

for column in nonnumeric_data.columns:

    print (nonnumeric_data[column].value_counts())



    

# Indequate strings that will be mapped to numeric values:



data = data.replace('positive', 1) # applies to all features

data = data.replace('negative', 0) # applies to all features

data = data.replace('identified', 1) # applies to all features

data = data.replace('not_identified', 0) # applies to all features

data = data.replace('absent', 0) # applies to all features

data = data.replace('present', 1) # applies to all features

data = data.replace('not_done', 0.5) # applies to all features. The test value could either be positive or negative. 



# We will assume, for 'not_done' that the probabilities of outcome of the test are uniform, that is, 

# that P(positive) = P(negative), as we don't have enough data (or time) to perform distribution estimation

# in order to acquire a better tool for guessing/approximating the real probabilities.



data = data.replace('detected', 1) # applies to all features

data = data.replace('not_detected', 0) # applies to all features

data = data.replace('Não Realizado', 0) # applies to all features

data = data.replace('normal', 1) # applies to all features

data['Urine - Aspect'] = data['Urine - Aspect'].astype('category')

data['Urine - Aspect'] = data['Urine - Aspect'].cat.codes # simple mapping suffices

data['Urine - Density'] = data['Urine - Density'].replace('normal', 1)

data['Urine - Sugar'] = data['Urine - Sugar'].replace('<1000', 500) # mean between 0 and first other category (1000)

data['Urine - Leukocytes'] = data['Urine - Leukocytes'].astype('category')

data['Urine - Leukocytes'] = data['Urine - Leukocytes'].cat.codes # simple mapping suffices

data['Urine - Yeasts'] = data['Urine - Yeasts'].astype('category')

data['Urine - Yeasts'] = data['Urine - Yeasts'].cat.codes # simple mapping suffices

data['Urine - Crystals'] = data['Urine - Crystals'].astype('category')

data['Urine - Crystals'] = data['Urine - Crystals'].cat.codes # simple mapping suffices

data['Urine - Color'] = data['Urine - Color'].astype('category')

data['Urine - Color'] = data['Urine - Color'].cat.codes # simple mapping suffices



# Resolving NaN values, feature-by-feature, for the nonnumeric_data columns

data['Urine - pH'] = data['Urine - pH'].fillna(0)

data['Urine - pH'] = data['Urine - pH'].astype('float64')



# simple mapping suffices is justified as follows: the data will be categorized without creating a bias, specifically, 

# as we will scale the data. Where 'scaling' is normalization and standardization of the data.

# The data has to be scaled in order to avoid creating the well known magnitude and centrality biases.
# Removing the patient's ID, and admittance data.



# The patient's ID carry no useful information, as individual points that might never repeat (the patient might never)

# be tested for COVID-19 again, do not generalize well.



# At this point in time, the patient has just arrive at the ER. There is no data on wether or not he was admitted

# to the general ward, or to any other facility.



del data['Patient ID']

del data['Patient addmited to regular ward (1=yes. 0=no)']

del data['Patient addmited to semi-intensive unit (1=yes. 0=no)']

del data['Patient addmited to intensive care unit (1=yes. 0=no)']
data = data.fillna(0)
data = data.loc[:, data.std(axis = 0) > 0]
target_feature = 'SARS-Cov-2 exam result'

features = data.loc[:, data.columns != target_feature]

target = data.loc[:, data.columns == target_feature]

target = target.to_numpy()

target = target.ravel()



features = preprocessing.scale(features)

linear_estimator_feature_importance = LinearSVC(C=0.02, penalty="l1", dual=False).fit(features, target)

svm_model = SelectFromModel(linear_estimator_feature_importance, prefit = True)

relevant_features = svm_model.transform(features)



print ('Basis of learning hyperspace has', relevant_features.shape[1], 'dimensions.')

print ('The reduction in use of features was', (relevant_features.shape[1]/features.shape[1])*100, '%.')

print ('From', features.shape[1], 'to', relevant_features.shape[1], 'features.')



# 1. Repeated training and testing, using probabilistic choosing of train and test data



# Storing experiment results, and setting its parameters

experiment_results = []

rounds_of_experiment = 33 # Gauss's theorem condition

test_proportion = 0.33 # 1/3 of the data should be used as test data, for each round of experiment

regularization_intensity = 1.0 # L2 penalization, the more distant from 1, the more agressive



# It is very important to notice that several hyper-parameter optimization experiments were conducted, and they have not

# presented improvements over the parameter values that are shown here.

# To the curious reader, the optimization method used was the Randomized search on hyper parameters.



for k in range (1, rounds_of_experiment):

    # Segmenting the data into training and testing data

    X_train, X_test, Y_train, Y_test = train_test_split(relevant_features, target, test_size = test_proportion)

    

    # Now that we know which dimensions are more important to the learning process, we can use them to train

    # a classifier that deals well with high dimensional sparse data, that might follow a non-linear process.

    # As both the disease synthomatic manifestations and its evolution can be non-linear over time.

    # For such task, the literature suggests a Support Vector Machine.

    

    SVM = SVC(C = regularization_intensity,

              kernel = 'rbf', 

              gamma = 'auto', 

              shrinking = False)

    

    SVM.fit(X_train,Y_train)

    predictions = SVM.predict(X_test)

    

    result = SVM.score(X_test, Y_test)

    experiment_results.append(result)

    

    print(classification_report(Y_test, predictions))



    

# The overall score is the mean of all the scores obtained, as the score values are real numbers, generated by

# a complex algorithm that operates on probabilistically chosen data, they are not very likely to repeat themselves.

# Therefore, for consistency, we choose to use the mean instead of the median.

print(' >> Overall accuracy with', rounds_of_experiment, ' rounds of experimentation:', mean(experiment_results))

# 2. Stratified K-fold cross-validation



SVM = SVC(C = regularization_intensity, kernel = 'rbf', gamma = 'auto', shrinking = False)



X = relevant_features

Y = target



skf = StratifiedKFold(n_splits=33)

for train, test in skf.split(X, Y):

    SVM.fit(X[train], Y[train])



print (SVM.score(X,Y))

print(classification_report(Y,SVM.predict(X)))

# 3. Single split into training and testing data



SVM = SVC(C = regularization_intensity, kernel = 'rbf', gamma = 'auto', shrinking = False)

SVM.fit(X,Y)



print (SVM.score(X,Y))

print(classification_report(Y,SVM.predict(X)))

data = pd.read_csv("../input/covid19-csv/dataset_csv.CSV", encoding ='latin1')

nonnumeric_data = data.select_dtypes(include='object')



for column in nonnumeric_data.columns:

    print (nonnumeric_data[column].value_counts())



# Indequate strings that will be mapped to numeric values

data = data.replace('positive', 1) # applies to all features

data = data.replace('negative', 0) # applies to all features

data = data.replace('identified', 1) # applies to all features

data = data.replace('not_identified', 0) # applies to all features

data = data.replace('absent', 0) # applies to all features

data = data.replace('present', 1) # applies to all features

data = data.replace('not_done', 0.5) # applies to all features.

data = data.replace('detected', 1) # applies to all features

data = data.replace('not_detected', 0) # applies to all features

data = data.replace('Não Realizado', 0) # applies to all features

data = data.replace('normal', 1) # applies to all features

data['Urine - Aspect'] = data['Urine - Aspect'].astype('category')

data['Urine - Aspect'] = data['Urine - Aspect'].cat.codes # simple mapping suffices

data['Urine - Density'] = data['Urine - Density'].replace('normal', 1)

data['Urine - Sugar'] = data['Urine - Sugar'].replace('<1000', 500) # mean between 0 and first other category (1000)

data['Urine - Leukocytes'] = data['Urine - Leukocytes'].astype('category')

data['Urine - Leukocytes'] = data['Urine - Leukocytes'].cat.codes

data['Urine - Yeasts'] = data['Urine - Yeasts'].astype('category')

data['Urine - Yeasts'] = data['Urine - Yeasts'].cat.codes

data['Urine - Crystals'] = data['Urine - Crystals'].astype('category')

data['Urine - Crystals'] = data['Urine - Crystals'].cat.codes

data['Urine - Color'] = data['Urine - Color'].astype('category')

data['Urine - Color'] = data['Urine - Color'].cat.codes



# Resolving NaN values, feature-by-feature, for the nonnumeric_data columns

data['Urine - pH'] = data['Urine - pH'].fillna(0)

data['Urine - pH'] = data['Urine - pH'].astype('float64')



# ------------------------------------------------------------------------------

# From this point on, this approach differs from the solution to the first task.

# ------------------------------------------------------------------------------



# Structuring the outcome of the patient's admittance (or he's lack thereof) as a multiclass problem

data['Patient addmited to regular ward (1=yes. 0=no)'] = data['Patient addmited to regular ward (1=yes. 0=no)'] .replace(1,1) 

data['Patient addmited to semi-intensive unit (1=yes. 0=no)'] = data['Patient addmited to semi-intensive unit (1=yes. 0=no)'].replace(1,2)

data['Patient addmited to intensive care unit (1=yes. 0=no)'] = data['Patient addmited to intensive care unit (1=yes. 0=no)'].replace(1,3)

data['target'] = data['Patient addmited to regular ward (1=yes. 0=no)'] + data['Patient addmited to semi-intensive unit (1=yes. 0=no)'] +                  data['Patient addmited to intensive care unit (1=yes. 0=no)']



# Removing the patient's ID, and admittance data

del data['Patient ID']

del data['SARS-Cov-2 exam result']

del data['Patient addmited to regular ward (1=yes. 0=no)']

del data['Patient addmited to semi-intensive unit (1=yes. 0=no)']

del data['Patient addmited to intensive care unit (1=yes. 0=no)']



data = data.fillna(0)
data = data.loc[:, data.std(axis = 0) > 0]



target_feature = 'target'

features = data.loc[:, data.columns != target_feature]

target = data.loc[:, data.columns == target_feature]

target = target.to_numpy()

target = target.ravel()



features = preprocessing.scale(features)

linear_estimator_feature_importance = LinearSVC(C=0.02, penalty="l1", dual=False).fit(features, target)

svm_model = SelectFromModel(linear_estimator_feature_importance, prefit = True)

relevant_features = svm_model.transform(features)



print ('Basis of learning hyperspace has', relevant_features.shape[1], 'dimensions.')

print ('The reduction in use of features was', (relevant_features.shape[1]/features.shape[1])*100, '%.')

print ('From', features.shape[1], 'to', relevant_features.shape[1], 'features.')

# 1. Repetead train and test



# Storing experiment results, and setting its parameters

experiment_results = []

rounds_of_experiment = 33 # Gauss's theorem condition

test_proportion = 0.33 # 1/3 of the data should be used as test data, for each round of experiment

regularization_intensity = 1.0 # L2 penalization, the more distant from 1, the more agressive



for k in range (1, rounds_of_experiment):

    # Segmenting the data into training and testing data

    X_train, X_test, Y_train, Y_test = train_test_split(relevant_features, target, test_size = test_proportion)

    

    # Now that we know which dimensions are more important to the learning process, we can use them to train

    # a classifier that deals well with high dimensional sparse data, that might follow a non-linear process.

    # As both the disease synthomatic manifestations and its evolution can be non-linear over time.

    # For such task, the literature suggests a Support Vector Machine.

    

    SVM = SVC(C = regularization_intensity,

              kernel = 'rbf', 

              gamma = 'scale', 

              shrinking = False,

              class_weight = 'balanced',

              decision_function_shape = 'ovr')

    

    SVM.fit(X_train,Y_train)

    predictions = SVM.predict(X_test)

    

    result = SVM.score(X_test, Y_test)

    experiment_results.append(result)

    

    print(classification_report(Y_test, predictions))



print(' >> Overall accuracy with', rounds_of_experiment, ' rounds of experimentation:', mean(experiment_results))

# 2. Stratified K-Fold cross-validation



SVM = SVC(C = regularization_intensity,

              kernel = 'rbf', 

              gamma = 'scale', 

              shrinking = False,

              class_weight = 'balanced',

              decision_function_shape = 'ovr')

    

X = relevant_features

Y = target



skf = StratifiedKFold(n_splits=33)

for train, test in skf.split(X, Y):

    SVM.fit(X[train], Y[train])



print (SVM.score(X,Y))

print(classification_report(Y,SVM.predict(X)))

# 3. Simple single train and test split



SVM = SVC(C = regularization_intensity,

              kernel = 'rbf', 

              gamma = 'scale', 

              shrinking = False,

              class_weight = 'balanced',

              decision_function_shape = 'ovr')  

SVM.fit(X,Y)



print(SVM.score(X,Y))

print(classification_report(Y,SVM.predict(X)))

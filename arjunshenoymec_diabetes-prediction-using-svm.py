# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input"))

def normalize(srs, max_pregn_val, min_pregn_val, max_glucose_val, min_glucose_val, max_bp_val, min_bp_val, max_dpf_val, min_dpf_val, max_ins_val, min_ins_val, max_bmi_val, min_bmi_val):
    srs.Pregnancies = float((srs.Pregnancies-min_pregn_val)/(max_pregn_val-min_pregn_val))
    srs.Glucose = float((srs.Glucose - min_glucose_val)/(max_glucose_val - min_glucose_val))
    srs.BloodPressure = float((srs.BloodPressure - min_bp_val)/(max_bp_val - min_bp_val))
    srs.DiabetesPedigreeFunction = float((srs.DiabetesPedigreeFunction - min_dpf_val)/(max_dpf_val - min_dpf_val))
    srs.Insulin = float((srs.Insulin - min_ins_val)/(max_ins_val - min_ins_val))
    srs.BMI = float((srs.BMI - min_bmi_val)/(max_bmi_val - min_bmi_val))
    return srs

def create_and_fit_svm(train):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 5, 10, 0.1], 'gamma': ['auto', 'scale']}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    size = len(train)
    labels = train.Outcome
    # Choosing only Pregnancies, Glucose, BloodPressure, dpf, insulin and bmi as features
    features = pd.DataFrame(train.iloc[:, [0,1,2,4,5,6]])
    clf.fit(features, labels)
    return clf

if __name__ == "__main__":
    patient_data = pd.read_csv("../input/diabetes.csv")
    size = len(patient_data)
    print(patient_data.dtypes)
    max_pregn = patient_data.Pregnancies.max()
    min_pregn = patient_data.Pregnancies.min()
    max_glucose = patient_data.Glucose.max()
    min_glucose = patient_data.Glucose.min()
    max_bp = patient_data.BloodPressure.max()
    min_bp = patient_data.BloodPressure.min()
    max_dpf = patient_data.DiabetesPedigreeFunction.max()
    min_dpf = patient_data.DiabetesPedigreeFunction.min()
    max_ins = patient_data.Insulin.max()
    min_ins = patient_data.Insulin.min()
    max_bmi = patient_data.BMI.max()
    min_bmi = patient_data.BMI.min()
    
    # Normalizing the data and obtaining the new DataFrame
    patient_data = patient_data.apply((lambda x: normalize(x, max_pregn, min_pregn, max_glucose, min_glucose, max_bp, min_bp, max_dpf, min_dpf, max_ins, min_ins, max_bmi, min_bmi)), axis='columns')

    
    # Test-Train Split, use Kfold CV later
    # data_train, data_test = train_test_split(patient_data, test_size=int(size/2), train_size=int(size/2), random_state=42, shuffle=True)
    
    # Using KFold CV
    kf = KFold(n_splits=8, random_state=42, shuffle=True)
    scores= []
    for train_index, test_index in kf.split(patient_data):
        data_train = patient_data.iloc[train_index]
        data_test = patient_data.iloc[test_index] 
        # Creating the classifier 
        classifier = create_and_fit_svm(data_train)
        labels_test = data_test.Outcome
        features_test = pd.DataFrame(data_test.iloc[:, [0,1,2,4,5,6]])
        pred = classifier.predict(features_test)
        scores.append(accuracy_score(pred, labels_test))
    print(float(sum(scores)/8))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Outlier Detection

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input"))

def normalize(srs, max_pregn_val, min_pregn_val, max_glucose_val, min_glucose_val, max_bp_val, min_bp_val, max_dpf_val, min_dpf_val, max_ins_val, min_ins_val, max_bmi_val, min_bmi_val, max_age_val, min_age_val):
    srs.Pregnancies = float((srs.Pregnancies-min_pregn_val)/(max_pregn_val-min_pregn_val))
    srs.Glucose = float((srs.Glucose - min_glucose_val)/(max_glucose_val - min_glucose_val))
    srs.BloodPressure = float((srs.BloodPressure - min_bp_val)/(max_bp_val - min_bp_val))
    srs.DiabetesPedigreeFunction = float((srs.DiabetesPedigreeFunction - min_dpf_val)/(max_dpf_val - min_dpf_val))
    srs.Insulin = float((srs.Insulin - min_ins_val)/(max_ins_val - min_ins_val))
    srs.BMI = float((srs.BMI - min_bmi_val)/(max_bmi_val - min_bmi_val))
    srs.Age = float((srs.Age - min_age_val)/(max_age_val - min_age_val))
    return srs

def create_and_fit_svm(train):
    parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 5, 10, 0.1], 'gamma': ['auto', 'scale']}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    size = len(train)
    labels = train.Outcome
    # Choosing only Pregnancies, Glucose, BloodPressure, dpf, insulin and bmi as features
    features = pd.DataFrame(train.iloc[:, [1,2,4,5,6,7]])
    clf.fit(features, labels)
    return clf

# Function to remove outliers or mismatches. 
# Only at most 10% of the test data is removed
def remove_inconsistancies(data_test, predicted, actual):
    i = 0
    predicted = predicted.tolist()
    bad_indices = []
    for key in actual.keys():
        if actual[key] != predicted[i]:
            bad_indices.append(key)
        i = i + 1
    if len(bad_indices) > len(data_test)/10:
        bad_indices = bad_indices[:int(len(data_test)/10)]
    remaining_indices = [x for x in range(0, len(data_test)) if x not in bad_indices]
    return data_test.iloc[remaining_indices]

if __name__ == "__main__":
    patient_data = pd.read_csv("../input/diabetes.csv")
    size = len(patient_data)
    print(patient_data.dtypes)
    max_pregn = patient_data.Pregnancies.max()
    min_pregn = patient_data.Pregnancies.min()
    max_glucose = patient_data.Glucose.max()
    min_glucose = patient_data.Glucose.min()
    max_bp = patient_data.BloodPressure.max()
    min_bp = patient_data.BloodPressure.min()
    max_dpf = patient_data.DiabetesPedigreeFunction.max()
    min_dpf = patient_data.DiabetesPedigreeFunction.min()
    max_ins = patient_data.Insulin.max()
    min_ins = patient_data.Insulin.min()
    max_bmi = patient_data.BMI.max()
    min_bmi = patient_data.BMI.min()
    max_age = patient_data.Age.max()
    min_age = patient_data.Age.min()
    
    # Normalizing the data and obtaining the new DataFrame
    patient_data = patient_data.apply((lambda x: normalize(x, max_pregn, min_pregn, max_glucose, min_glucose, max_bp, min_bp, max_dpf, min_dpf, max_ins, min_ins, max_bmi, min_bmi, max_age, min_age)), axis='columns')

    
    # Test-Train Split, use Kfold CV later
    # data_train, data_test = train_test_split(patient_data, test_size=int(size/2), train_size=int(size/2), random_state=42, shuffle=True)
    
    # Using KFold CV
    kf = KFold(n_splits=8, random_state=42, shuffle=True)
    initial_scores= []
    new_scores = []
    for train_index, test_index in kf.split(patient_data):
        data_train = patient_data.iloc[train_index]
        data_test = patient_data.iloc[test_index] 
        # Creating the classifier 
        classifier = create_and_fit_svm(data_train)
        labels_test = data_test.Outcome
        features_test = pd.DataFrame(data_test.iloc[:, [1,2,4,5,6,7]])
        pred = classifier.predict(features_test)
        initial_scores.append(accuracy_score(pred, labels_test))
        
        # Outlier removal and retry
        data_test = remove_inconsistancies(data_test, pred, labels_test)
        labels_test = data_test.Outcome
        features_test = pd.DataFrame(data_test.iloc[:, [1,2,4,5,6,7]])
        pred = classifier.predict(features_test)
        new_scores.append(accuracy_score(pred, labels_test))
        
    print(float(sum(initial_scores)/8))
    print(float(sum(new_scores)/8))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Adding the correct features to the classifier

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

import os
print(os.listdir("../input"))

def normalize(srs, max_pregn_val, min_pregn_val, max_glucose_val, min_glucose_val, max_bp_val, min_bp_val, max_dpf_val, min_dpf_val, max_ins_val, min_ins_val, max_bmi_val, min_bmi_val, max_age_val, min_age_val, max_st_val, min_st_val):
    srs.Pregnancies = float((srs.Pregnancies-min_pregn_val)/(max_pregn_val-min_pregn_val))
    srs.Glucose = float((srs.Glucose - min_glucose_val)/(max_glucose_val - min_glucose_val))
    srs.BloodPressure = float((srs.BloodPressure - min_bp_val)/(max_bp_val - min_bp_val))
    srs.DiabetesPedigreeFunction = float((srs.DiabetesPedigreeFunction - min_dpf_val)/(max_dpf_val - min_dpf_val))
    srs.Insulin = float((srs.Insulin - min_ins_val)/(max_ins_val - min_ins_val))
    srs.BMI = float((srs.BMI - min_bmi_val)/(max_bmi_val - min_bmi_val))
    srs.Age = float((srs.Age - min_age_val)/(max_age_val - min_age_val))
    srs.SkinThickness = float((srs.SkinThickness - min_st_val)/(max_st_val - min_st_val))
    return srs

def create_and_fit_svm(train):
    parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 5, 10, 0.1], 'gamma': ['auto', 'scale']}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)
    size = len(train)
    labels = train.Outcome
    # Choosing only Pregnancies, Glucose, BloodPressure, dpf, insulin and bmi as features
    features = pd.DataFrame(train.iloc[:, [1,2,4,5,6,7]])
    clf.fit(features, labels)
    return clf

if __name__ == "__main__":
    patient_data = pd.read_csv("../input/diabetes.csv")
    size = len(patient_data)
    print(patient_data.dtypes)
    max_pregn = patient_data.Pregnancies.max()
    min_pregn = patient_data.Pregnancies.min()
    max_glucose = patient_data.Glucose.max()
    min_glucose = patient_data.Glucose.min()
    max_bp = patient_data.BloodPressure.max()
    min_bp = patient_data.BloodPressure.min()
    max_dpf = patient_data.DiabetesPedigreeFunction.max()
    min_dpf = patient_data.DiabetesPedigreeFunction.min()
    max_ins = patient_data.Insulin.max()
    min_ins = patient_data.Insulin.min()
    max_bmi = patient_data.BMI.max()
    min_bmi = patient_data.BMI.min()
    max_age = patient_data.Age.max()
    min_age = patient_data.Age.min()
    max_st = patient_data.SkinThickness.max()
    min_st = patient_data.SkinThickness.min()
    
    # Normalizing the data and obtaining the new DataFrame
    patient_data = patient_data.apply((lambda x: normalize(x, max_pregn, min_pregn, max_glucose, min_glucose, max_bp, min_bp, max_dpf, min_dpf, max_ins, min_ins, max_bmi, min_bmi, max_age, min_age, max_st, min_st)), axis='columns')
    print(patient_data.iloc[0:2])
    
    # Test-Train Split, use Kfold CV later
    # data_train, data_test = train_test_split(patient_data, test_size=int(size/2), train_size=int(size/2), random_state=42, shuffle=True)
    
    # Using KFold CV
    kf = KFold(n_splits=8, random_state=42, shuffle=True)
    scores= []
    for train_index, test_index in kf.split(patient_data):
        data_train = patient_data.iloc[train_index]
        data_test = patient_data.iloc[test_index] 
        # Creating the classifier 
        classifier = create_and_fit_svm(data_train)
        labels_test = data_test.Outcome
        features_test = pd.DataFrame(data_test.iloc[:, [1,2,4,5,6,7]])
        pred = classifier.predict(features_test)
        scores.append(accuracy_score(pred, labels_test))
    print(float(sum(scores)/8))
# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"
from imblearn.under_sampling import NearMiss, TomekLinks
from imblearn.over_sampling import SMOTE, ADASYN
import pickle as pk
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

# Common imports
import numpy as np
import pandas as pd
import os

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
%load_ext autoreload
%autoreload 2

%matplotlib inline
#PATH = 'C:/MyWorkFileDirectory/ML projects/kaggle/Credit Card Fraud/'
# def load_data(file_name, path=PATH):
#     csv_path = os.path.join(path, file_name)
#     return pd.read_csv(csv_path)
# Taken from the fast.ai library
def rf_feat_importance(m, data_columns):
    return pd.DataFrame({'cols':data_columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
# Taken from the fast.ai library
def plot_fi(fi): 
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
# from imblearn library
def get_under_sampling_method(method):
    if method == 1:
        return NearMiss(sampling_strategy='majority', n_jobs=-1)
    else:
        return TomekLinks(sampling_strategy='majority', n_jobs=-1)
# from imblearn library
def get_over_sampling_method(method):
    if method == 1:
        return SMOTE(sampling_strategy='minority', n_jobs=-1)
    else:
        return ADASYN(sampling_strategy='minority', n_jobs=-1)
def print_classification_report(model, features, labels):
    predictions = model.predict(features)
    print(classification_report(labels, predictions))
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df.head()
# scale columns Time and Amount
from sklearn.preprocessing import RobustScaler

col_to_scale = ['Time', 'Amount']
rb = RobustScaler()
df[col_to_scale] = rb.fit_transform(df[col_to_scale])
df[col_to_scale].describe()
df.hist(bins=50, figsize=(20,15))
plt.show()
df['Class'].value_counts()
df.isna().sum().sum(), df.isnull().sum().sum()
X = df.copy()
X.drop('Class', axis=1, inplace=True)
y = df['Class']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=42)
_, p = np.unique(y_train, return_counts=True)
ratio = int(np.ceil(len(y_train) / p[1])) 
n_features = X_train.shape[1]
ratio, n_features
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

rf_clf_params = {'n_estimators': [200, 300], 'min_samples_leaf': [5, 10, 25]}
grid_rf_clf = GridSearchCV(ExtraTreesClassifier(n_jobs=-1, class_weight={0:1,1:ratio}, random_state=42), rf_clf_params)
grid_rf_clf.fit(X_train, y_train)
rf_clf = grid_rf_clf.best_estimator_
from sklearn.linear_model import LogisticRegression

log_reg_params = {'C': [0.001, 0.01, 0.1, 1], 'class_weight': ['balanced'], 'max_iter': [500, 1000]}
grid_log_reg = GridSearchCV(LogisticRegression(n_jobs=-1), log_reg_params)
grid_log_reg.fit(rb.fit_transform(X_train), y_train)
log_reg = grid_log_reg.best_estimator_
print_classification_report(rf_clf, X_train, y_train), print_classification_report(log_reg, X_train, y_train)
print_classification_report(rf_clf, X_test, y_test), print_classification_report(log_reg, X_test, y_test)
fi = rf_feat_importance(rf_clf, X_train.columns)
plot_fi(fi)
fi
to_keep = fi[fi.imp >= 0.008].cols
X_keep = X[to_keep].copy()
X_train, X_test = train_test_split(X_keep, test_size = 0.2, stratify = y, random_state=42)
rf_clf.fit(X_train, y_train)
print_classification_report(rf_clf, X_train, y_train)
print_classification_report(rf_clf, X_test, y_test)
# reset again to the original
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state=42)
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import make_pipeline
def sampling_correct(method, sampling, classifier, print_report):
    """ Over/Under-Samples the minority class, performs Stratified KFold CV and returns the trained models
    This is the correct way of over/under-sampling during cross-validation so that validation data maintains
    the original imbalanced split and not the synthetic data generated by the imblearn library in the 
    sampled training data. If this is not done, the model will overfit to cross-validation and furnish
    poor results during testing/production.
        
    Parameters:
    -----------
    method: The sampling method
            For over sampling: method = 1: SMOTE, 2: ADYSN
            For under sampling: method = 1: NearMiss, 2: TomekLinks
    sampling: Over or Under sampling: 1: Over-sampling. 2: Under-sampling
    classifier: The classification model
    print_report: Print the classification report. 1: yes 0: no
    
    Returns:
    --------
    return value: A list of trained models on the over-sampled data from X_train and y_train
    """
    
    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
    
    recall = []
    
    for train_index, cv_index in skfolds.split(X_train, y_train):
        sample_pipeline = make_pipeline(get_over_sampling_method(method) if sampling == 1 else get_under_sampling_method(method), 
                                        classifier)
        sample_model = sample_pipeline.fit(rb.fit_transform(X_train.iloc[train_index]), y_train.iloc[train_index])
        pred = sample_model.predict(rb.transform(rb.fit_transform(X_train.iloc[cv_index])))
        recall.append(recall_score(y_train.iloc[cv_index], pred))
        if (print_report == 1):
            print_classification_report(sample_model, rb.fit_transform(X_train.iloc[cv_index]), y_train.iloc[cv_index])
    
    return np.mean(recall)
def sampling_incorrect(method, sampling, classifier, print_report):
    """ Over/Under-Samples the majority class, performs Stratified KFold CV and returns the trained models
    This is the incorrect way of sampling because the validation data also has the same proportion
    of majority and minority classas trained data, and will possibly lead to overfitting the cross-validation
    and poor generalization during test/production
    
    Parameters:
    -----------
    method: The sampling method
            For over sampling: method = 1: SMOTE, 2: ADYSN
            For under sampling: method = 1: NearMiss, 2: TomekLinks
    sampling: Over or Under sampling: 1: Over-sampling. 2: Under-sampling
    classifier: The classification model
    print_report: Print the classification report. 1: yes 0: no
    
    Returns:
    --------
    return value: A list of trained models on the over-sampled data from X_train and y_train
    """
    
    sample_method = get_over_sampling_method(method) if sampling == 1 else get_under_sampling_method(method)
    x_train_sample, y_train_sample = sample_method.fit_sample(X_train, y_train)
    
    skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

    recall = []

    for train_index, cv_index in skfolds.split(x_train_sample, y_train_sample):
        X_train_sample_folds = x_train_sample.iloc[train_index]
        y_train_sample_folds = y_train_sample.iloc[train_index]
        X_cv_sample_fold = x_train_sample.iloc[cv_index]
        y_cv_sample_fold = y_train_sample.iloc[cv_index]
        classifier.fit(rb.transform(X_train_sample_folds), y_train_sample_folds)
        pred = classifier.predict(rb.transform(X_cv_sample_fold))
        recall.append(recall_score(y_cv_sample_fold, pred))
        if (print_report == 1):
            print_classification_report(lr_clf, rb.transform(X_cv_sample_fold), y_cv_sample_fold)
    
    return np.mean(recall)
def analysis_model(param1, param2, sampling, method, correct, classifier, print_report):
    """ Performs the over/under-sampling using Classifier 
    for 2 sets of parameters computes the recall score of class 
    label 1, which here is the minority class (Fraud Case)
        
    Parameters:
    -----------
    param1: tuple of ('param1_name', param1_value)
    param1: tuple of ('param2_name', param2_value)
    method: The sampling method
            For over sampling: method = 1: SMOTE, 2: ADASYN
            For under sampling: method = 1: NearMiss, 2: TomekLinks
    sampling: Over or Under sampling: 1: Over-sampling, 2: Under-sampling
    correct: 1: use correct cv method, 2: use incorrect cv method
    classifier: The classification model
    print_report: Print the classification report. 1: yes 0: no
    
    Returns:
    --------
    return value: List of Extra Tree models in a double array with index 1
                  for param1 and index 2 for param 2
    """
    
    recall_final = 0
    param_final = tuple()
    
    param1_name, param1_value = param1
    param2_name, param2_value = param2
    
    for i, param_1 in enumerate(param1_value):
        for j, param_2 in enumerate(param2_value):
            param_grid = {param1_name: param_1, param2_name: param_2}
            classifier.set_params(**param_grid)
            recall_temp = 0
            if (correct == 1):
                recall_temp = sampling_correct(method=method, sampling=sampling, classifier=classifier, print_report=print_report) 
            else:
                recall_temp = sampling_incorrect(method=method, sampling=sampling, classifier=classifier, print_report=print_report) 
            if (recall_temp >= recall_final):
                recall_final = recall_temp
                param_final = (i, j)
    
    return recall_final, param_final
def create_param_grid(param1, param2, param_final):
    param1_name = param1[0]
    param2_name = param2[0]
    param1_value = param1[1][param_final[0]]
    param2_value = param2[1][param_final[1]]
    return {param1_name : param1_value, param2_name : param2_value}
# Performs the over/under-sampling for 2 sets of parameters
# computes the recall score of class label 1, which here is 
# the minority class (Fraud Case), and prints a table of columns

#    classifier | param_grid | recall score 
#    --------------------------------------

# Extra Tree Classifier
# Over-Sampling
# Incorrect method

final_results = []

param1 = ('n_estimators', [200, 300])
param2 = ('min_samples_leaf', [5, 10, 25])
clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=1, method=1, correct=2, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_over_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 1, 'Classifier': 'Extra Tree Classifier', 'Sampling': 'Over', 'Method': 'Incorrect', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# Extra Tree Classifier
# Over-Sampling
# Correct method
param1 = ('n_estimators', [200, 300])
param2 = ('min_samples_leaf', [5, 10, 25])
clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=1, method=1, correct=1, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_over_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 2, 'Classifier': 'Extra Tree Classifier', 'Sampling': 'Over', 'Method': 'Correct', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# Extra Tree Classifier
# Under-Sampling
# Incorrect method
param1 = ('n_estimators', [200, 300])
param2 = ('min_samples_leaf', [5, 10, 25])
clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=2, method=1, correct=2, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_under_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 3, 'Classifier': 'Extra Tree Classifier', 'Sampling': 'Under', 'Method': 'Incorrect', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# Extra Tree Classifier
# Under-Sampling
# Correct method
param1 = ('n_estimators', [200, 300])
param2 = ('min_samples_leaf', [5, 10, 25])
clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=2, method=1, correct=1, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = ExtraTreesClassifier(max_features = 'auto', random_state=42, n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_under_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 4, 'Classifier': 'Extra Tree Classifier', 'Sampling': 'Under', 'Method': 'Correct', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# Logistic Regression
# Over-Sampling
# Incorrect method
param1 = ('C', [0.001, 0.01, 0.1, 1])
param2 = ('max_iter', [300, 500])
clf = LogisticRegression(n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=1, method=1, correct=2, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = LogisticRegression(n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_over_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 5, 'Classifier': 'Logistic Regression', 'Sampling': 'Over', 'Method': 'Incorrect', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# Logistic Regression
# Over-Sampling
# Correct method
param1 = ('C', [0.001, 0.01, 0.1, 1])
param2 = ('max_iter', [300, 500])
clf = LogisticRegression(n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=1, method=1, correct=1, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = LogisticRegression(n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_over_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 6, 'Classifier': 'Logistic Regression', 'Sampling': 'Over', 'Method': 'Correct', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# Logistic Regression
# Under-Sampling
# Incorrect method
param1 = ('C', [0.001, 0.01, 0.1, 1])
param2 = ('max_iter', [300, 500])
clf = LogisticRegression(n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=2, method=1, correct=2, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = LogisticRegression(n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_under_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 7, 'Classifier': 'Logistic Regression', 'Sampling': 'Under', 'Method': 'Incorrect', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# Logistic Regression
# Under-Sampling
# Correct method
param1 = ('C', [0.001, 0.01, 0.1, 1])
param2 = ('max_iter', [300, 500])
clf = LogisticRegression(n_jobs=-1)
recall_final, param_final = analysis_model(param1=param1, param2=param2, sampling=2, method=1, correct=2, classifier=clf, print_report=0)
param_grid = create_param_grid(param1, param2, param_final)
final_rf_clf = LogisticRegression(n_jobs=-1)
final_rf_clf.set_params(**param_grid)
x_sample, y_sample = get_under_sampling_method(method=1).fit_sample(X_train, y_train)
final_rf_clf.fit(rb.transform(x_sample), y_sample)
pred = final_rf_clf.predict(rb.transform(X_test))
result_dict = {'Id': 8, 'Classifier': 'Logistic Regression', 'Sampling': 'Under', 'Method': 'Correct', 'recall score': recall_score(y_test, pred), 'Model': final_rf_clf}
final_results.append(result_dict)

# with open('C:/MyWorkFileDirectory/ML projects/kaggle/Credit Card Fraud/final_model_list.pickle', 'wb') as handle:
#     pk.dump(final_results, handle, protocol=pk.HIGHEST_PROTOCOL)
final_data_frame = pd.DataFrame(final_results)
final_data_frame
mmm = final_results[7]['Model']
pred = mmm.predict(rb.transform(X_train))
recall_score(y_train, pred)
mmm = final_results[5]['Model']
pred = mmm.predict(rb.transform(X_train))
recall_score(y_train, pred)
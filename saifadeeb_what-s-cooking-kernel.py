import time

import pandas as pd
import json
import pdb

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import re
from nltk.corpus import wordnet as wn

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
def open_json_file(file):
    """Function to open JSON files"""
    json_opened = json.load(open(file))
    return json_opened
#readining in data
train = open_json_file('../input/train.json')
test = open_json_file('../input/test.json')
print("--- sample of training data set ---")
train[0:1]
print("--- sample of testing data set ---")
test[0:1]
print("Training data set size:",len(train))
print("Testing data set size:",len(test))
def pre_process_data(data):
    """This function removes punctuations, special characters letters, and symbols from a list of words"""
    new_data = []
    for recipe in data:
        new_recipe = []
        for ingredient in recipe:
            new_ingredient_list = []
            for word in ingredient.split():
                word = re.sub('[^a-zA-Z -]+', '', word) # only keeping the letters, spaces, and hyphens
                new_ingredient_list.append(wn.morphy(word.lower().strip(",.!:?;' ")) or word.strip(",.!:?;' ")) # strip, stem, and append the word
            new_recipe.append(' '.join(new_ingredient_list))
        new_data.append(new_recipe)
    return new_data
#ingredients for training
x_train = list([train[i]['ingredients'] for i in range(len(train))])
#labels/cuisines for training
y_train_labels = [train[i]['cuisine'] for i in range(len(train))]
#ingredients for test data
x_test = list([test[i]['ingredients'] for i in range(len(test))])
#pre-process data
new_x_train = pre_process_data(x_train)
new_x_test = pre_process_data(x_test)
# prepare X and y
print ("--- Label Encode the Target Variable ---")
lb_enc = LabelEncoder()
y_enc_labels = lb_enc.fit_transform(y_train_labels)
print(y_enc_labels)
# Text Data Features
def bag_of_words(data):
	text_data = [' '.join(recipe).lower() for recipe in data] 
	return text_data 

def concatenated_words(data):
	text_data = [' '.join(word.replace(" ","_").lower() for word in recipe) for recipe in data]
	return text_data 
print ("--- Preparing text data ---")
#feature extraction - take #1 
train_text = bag_of_words(x_train)
submission_text = bag_of_words(x_test)

#feature extraction - take #2
### Here, we are treating every word as a feature
prep_train_text = bag_of_words(new_x_train)
prep_submission_text = bag_of_words(new_x_test)

#feature extraction - take #3
### If an ingredients has multiple words, we will be joining them with an underscore before seperating each ingredient.
prep_train_text_underscore = concatenated_words(new_x_train)
prep_submission_text_underscore = concatenated_words(new_x_test)


# Feature Engineering 
tfidf_enc = TfidfVectorizer()

def tf_idf_features(text, flag):
    """Fitting TFIDF vectorizer to training data and transforming everything else"""
    if flag == "train":
        x = tfidf_enc.fit_transform(text)
    else:
        x = tfidf_enc.transform(text)
    x = x.astype('float16')
    return x 
#creating features for each feature engineering approach
train_text_features = tf_idf_features(train_text, flag="train")
submission_text_features = tf_idf_features(submission_text, flag="submission")

prep_train_text_features = tf_idf_features(prep_train_text, flag="train")
prep_submission_text_features = tf_idf_features(prep_submission_text, flag="submission")

prep_train_text_underscore_features = tf_idf_features(prep_train_text_underscore, flag="train")
prep_submission_text_underscore_features = tf_idf_features(prep_submission_text_underscore, flag="submission")
#getting a benchmark
RANDOM_SEED = 1
names = ['DecisionTree','RandomForestClassifier','Logistic_Regression', "SVC"]

#declaring the necessary information for each regression model
regressors = [DecisionTreeClassifier(random_state = RANDOM_SEED),
    #RandomForestClassifier(n_estimators = 1, random_state = RANDOM_SEED), 
              RandomForestClassifier(n_estimators = 10, random_state = RANDOM_SEED), 
              LogisticRegression(random_state = RANDOM_SEED),
              SVC(C=10, gamma = 1, decision_function_shape=None, random_state = RANDOM_SEED),
              ]

def kfold_cross_validation(X,y, col_names):
    # In[315]:

    #ten cross-validation employed here

    #shuffling the data
    #np.random.seed(RANDOM_SEED)

    # specify the k-fold cross-validation design
    N_FOLDS = 3

    # set up numpy array for storing results
    cv_results = np.zeros((N_FOLDS, len(names)))

    kf = KFold(n_splits = N_FOLDS, shuffle = False, random_state = RANDOM_SEED)

    index_for_fold = 0 #fold count initialized

    for train_index, test_index in kf.split(X,y):
        print('\nFold index:', index_for_fold + 1,
             '------------------------------------------')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('\nShape of input data for this fold:',
              '\nData Set: (Observations, Variables)')
        print('X_train:', X_train.shape)
        print('X_test:',X_test.shape)
        print('y_train:', y_train.shape)
        print('y_test:',y_test.shape)

        index_for_method = 0 #initialize
        for name, reg_model in zip(names, regressors):
            print('\nRegression model evaluation for:', name)
            print(' Scikit Learn method:', reg_model)
            reg_model.fit(X_train, y_train) #fit on the train set for this fold

            #evaluate on the test set for this fold
            y_test_predict = reg_model.predict(X_test)

            fold_method_result = reg_model.score(X_test, y_test)

            cv_results[index_for_fold, index_for_method] = fold_method_result
            index_for_method += 1

        index_for_fold += 1
        cv_results_df = pd.DataFrame(cv_results)
        cv_results_df.columns = col_names
    return cv_results_df

import time
starttime = time.monotonic()

print("--- Performing 3-fold cross validation ---\n")

print("---------Standard Results---------")
train_text_cv = kfold_cross_validation(train_text_features,
                                       y_enc_labels, 
                                       col_names = ["DecisionTree",
                                                    "RandomForest", 
                                                    "LogisticRegression",
                                                    "SVM"])

print("\n\n---------Removing Special Characters & Punctuations Results---------")
prep_train_text_cv = kfold_cross_validation(prep_train_text_features,
                                            y_enc_labels,
                                            col_names = ["DecisionTree",
                                                         "RandomForest",
                                                         "LogisticRegression",
                                                         "SVM"])

print("\n\n---------Concatenated Words Results---------")
prep_train_text_underscore_cv = kfold_cross_validation(prep_train_text_underscore_features,
                                                       y_enc_labels,
                                                      col_names = ["DecisionTree",
                                                                   'RandomForest',
                                                                   'LogisticRegression',
                                                                   "SVM"])

print("\n\nThat took ", (time.monotonic()-starttime)/60, " minutes")
print("--- Standard Scores ---")
print(train_text_cv.mean())
print()
print("--- Removing Special Characters & Punctuations Scores ---")
print(prep_train_text_cv.mean())
print()
print("--- Concatenated Words Scores ---")
print(prep_train_text_underscore_cv.mean())
#chose the method of removing special characters and punctuations based on performance
baseline_models = pd.DataFrame(prep_train_text_cv.mean(),columns=["Avg Accuracy"])
baseline_models = baseline_models.reset_index()
baseline_models.columns = ["Model","Baseline Score"]
baseline_models
def grid_search_function(func_X_train, func_X_test, func_y_train, func_y_test, parameters, model):
    model = model(random_state=RANDOM_SEED)

    grid_search = GridSearchCV(model, parameters)

    classifier= grid_search.fit(func_X_train,func_y_train)

    #score = classifier.score(func_X_test, func_y_test)
    return classifier
def train_test_split_function(X,y, test_size_percent):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_percent, random_state=RANDOM_SEED)
    return X_train, X_test, y_train, y_test
### RANDOM FOREST CLASSIFIER
starttime = time.monotonic()
parameters = {'n_estimators':[10,100,300]}
X_train, X_test, y_train, y_test = train_test_split_function(prep_train_text_features, 
                                                             y_enc_labels, 
                                                             test_size_percent = 0.20)

rf_classifier = grid_search_function(X_train, X_test, y_train, y_test, 
                                     parameters, 
                                     model = RandomForestClassifier)

print("\n\nThat took ", (time.monotonic()-starttime)/60, " minutes")
rf_classifier.best_estimator_
def training_info_and_classification_report(X,y,test_size_percent, model):
    """This function retrieves the training and testing score from a model and calculates the
       classification report. The function will output the training score, testing score,
       classification report, confusion matrix array, normalized confusion matrix array, and
       a dataframe of the normalized confusion matrix. This dataframe can then be used to plot
       a confusion matrix error plot."""
    starttime = time.monotonic()
    X_train, X_test, y_train, y_test = train_test_split_function(X,y, test_size_percent)
    print("\n--- Fitting Model ---")
    model.fit(X_train, y_train)
    print("\n--- Predicting Cuisines ---")
    y_predict = model.predict(X_test)
    print("\n--- Scoring Model ---")
    model_training_score = model.score(X_train,y_train)
    
    model_testing_score = model.score(X_test,y_test)
    print("\n--- Creating Classification Report ---")
    clf_report = classification_report(y_test,
                          y_predict,
                          target_names = lb_enc.inverse_transform(model.classes_).tolist())
    
    array = confusion_matrix(y_test, y_predict)
    
    row_sums = array.sum(axis=1,keepdims=True)
    norm_conf_mx = array/row_sums

    error_matrix_df = pd.DataFrame(norm_conf_mx, index = [i for i in lb_enc.inverse_transform(model.classes_).tolist()],
                  columns = [i for i in lb_enc.inverse_transform(model.classes_).tolist()])
    
    print("\n\nThat took ", (time.monotonic()-starttime)/60, " minutes")
    
    return model_training_score, model_testing_score, clf_report, array, norm_conf_mx, error_matrix_df
training_score_rf,testing_score_rf,rf_classification_report, rf_array, rf_norm_conf_mx, rf_error_matrix_df = training_info_and_classification_report(prep_train_text_features,
                                                                                                     y_enc_labels,
                                                                                                     test_size_percent = 0.20,
                                                                                                     model = rf_classifier.best_estimator_)
print("           Random Forest Classification Report           ")
print(rf_classification_report)
plt.figure(figsize = (20,15))
sns.heatmap(rf_error_matrix_df, annot=False, fmt='g',cmap="Greens")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix Error Plot")
plt.show()
starttime = time.monotonic()
parameters = {'C':[1,5,10,100,1000]}
X_train, X_test, y_train, y_test = train_test_split_function(prep_train_text_features, 
                                                             y_enc_labels, 
                                                             test_size_percent = 0.20)

logreg_classifier = grid_search_function(X_train, X_test, y_train, y_test, parameters, model = LogisticRegression)
print("\n\nThat took ", (time.monotonic()-starttime)/60, " minutes")
logreg_classifier.best_estimator_
training_score_lgr,testing_score_lgr,lgr_classification_report, lgr_array, lgr_norm_conf_mx, lgr_error_matrix_df = training_info_and_classification_report(prep_train_text_features,
                                                                                                     y_enc_labels,
                                                                                                     test_size_percent = 0.20,
                                                                                                     model = logreg_classifier.best_estimator_)
print("           Logistic Regression Classification Report           ")
print(lgr_classification_report)
plt.figure(figsize = (20,15))
sns.heatmap(rf_error_matrix_df, annot=False, fmt='g',cmap="Greens")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix Error Plot")
plt.show()
###Took ~8 Hours to run this GridSearch
#starttime = time.monotonic()
#
#parameters = {'C':(1,10,100,300), 'decision_function_shape':[None], 'gamma': (0.01,1,2,3,'auto'), 'kernel':('rbf','poly','linear') }
#
#X_train, X_test, y_train, y_test = train_test_split_function(prep_train_text_features, 
#                                                             y_enc_labels, 
#                                                             test_size_percent = 0.20)
#
#all_svm_classifier = grid_search_function(X_train, X_test, y_train, y_test, parameters, model = SVC)
#
#print("\n\nThat took ", (time.monotonic()-starttime)/60, " minutes")
#
#all_svm_classifier.best_estimator_

##winning parameters: (C=1, decision_function_shape = None, gamma= 1, kernel='rbf')
svm_clf = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=1, kernel='rbf',
  max_iter=-1, probability=False, random_state=1, shrinking=True,
  tol=0.001, verbose=False)
svm_ovr = OneVsRestClassifier(svm_clf)
def get_training_info(model,full_X_train, full_y_train):
    """Seperate function to run for SVM due to lengthy processing time and conflict when probability = True.
       This function will retunr the training and testing score of a predefined model"""
    X_train, X_test, y_train, y_test = train_test_split_function(full_X_train, 
                                                             full_y_train, 
                                                             test_size_percent = 0.20)
    print("\n--- Fitting Model ---")
    model.fit(X_train, y_train)
    
    print("\n--- Scoring Model ---")
    model_training_score = model.score(X_train,y_train)
    model_testing_score = model.score(X_test,y_test)
    
    return model_training_score,model_testing_score
    
def get_classification_report(X,y,test_size_percent, model):
    """Seperate function to run for SVM due to lengthy processing time and conflict when probability = True.
       This function will calculate the classification report, classifcation matrix array, normalized classificaiton
       matrix array, and dataframe for the ormalized classificaiton matrix array."""
    X_train, X_test, y_train, y_test = train_test_split_function(X,y, test_size_percent)
    
    model.fit(X_train, y_train)
    
    y_predict = model.predict(X_test)
    
    clf_report = classification_report(y_test,
                          y_predict,
                          target_names = lb_enc.inverse_transform(model.classes_).tolist())
    
    array = confusion_matrix(y_test, y_predict)
    
    row_sums = array.sum(axis=1,keepdims=True)
    norm_conf_mx = array/row_sums

    error_matrix_df = pd.DataFrame(norm_conf_mx, index = [i for i in lb_enc.inverse_transform(model.classes_).tolist()],
                  columns = [i for i in lb_enc.inverse_transform(model.classes_).tolist()])
    
    return clf_report, array, norm_conf_mx, error_matrix_df
training_score_svm, testing_score_svm = get_training_info(svm_ovr,prep_train_text_features,y_enc_labels)     
#need 'probability = True' for OneVsRestClassifier VotingClassifier
p_svm_clf = SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma=1, kernel='rbf',
  max_iter=-1, probability=True, random_state=1, shrinking=True,
  tol=0.001, verbose=False) 
#svm_ovr = OneVsRestClassifier(all_svm_classifier.best_estimator_)
svm_ovr_clf = OneVsRestClassifier(p_svm_clf)
svm_classification_report, svm_array, svm_norm_conf_mx, svm_error_matrix_df = get_classification_report(prep_train_text_features,
                                                                                                     y_enc_labels,
                                                                                                     test_size_percent = 0.20,
                                                                                                     model = svm_ovr_clf)
print("           SVM Classification Report           ")
print(svm_classification_report)
plt.figure(figsize = (20,15))
sns.heatmap(svm_error_matrix_df, annot=False, fmt='g',cmap="Greens")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title("Confusion Matrix Error Plot")
plt.show()
rf_clf = rf_classifier.best_estimator_
lgr_clf = logreg_classifier.best_estimator_
#svm_clf = all_svm_classifier.best_estimator_

print(rf_clf,"\n")
print(lgr_clf,"\n")
print(svm_clf,"\n")
#starttime = time.monotonic()
#print("Getting best model results for Random Forest classifier")
#training_score_rf,testing_score_rf=get_training_info(rf_clf,prep_train_text_features,y_enc_labels)
#print("\nThat took ", (time.monotonic()-starttime)/60, " minutes\n")
#
#starttime = time.monotonic()
#print("Getting best model results for Logistic Regression")
#training_score_lgr,testing_score_lgr=get_training_info(lgr_clf,prep_train_text_features,y_enc_labels)
#print("\nThat took ", (time.monotonic()-starttime)/60, " minutes\n")
#
#starttime = time.monotonic()
#print("Getting best model results for SVM")
###Already retrieved##
##svm_ovr_clf = OneVsRestClassifier(svm_clf)
##training_score_svm,testing_score_svm=get_training_info(svm_ovr_clf,prep_train_text_features,y_enc_labels)
#print("\nThat took ", (time.monotonic()-starttime)/60, " minutes\n")
model_names = [
    "Random Forest Classifier",
    "Logistic Regression",
    "SVM",
    "VotingClassifier"]


best_model_testing_score = [
    testing_score_rf,
    testing_score_lgr,
    testing_score_svm,
    "-"
                   ]

best_model_training_score = [
    training_score_rf,
    training_score_lgr,
    training_score_svm,
    "-"
]
def model_to_submission_file(model, full_X_train, full_y_train, full_X_test ,nameOfcsvfile):
    print("\n--- Fitting Model ---")
    starttime = time.monotonic()
    model_clf = model.fit(full_X_train,full_y_train)
    
    print("\n--- Predicting Cuisines ---")
    submission_pred = model_clf.predict(full_X_test)
    test_cuisine = lb_enc.inverse_transform(submission_pred)

    test_id = [recipe['id'] for recipe in test]

    submission_df = pd.DataFrame({'id':test_id, 'cuisine':test_cuisine},columns = ['id','cuisine'])

    submission_df.to_csv('{}'.format(nameOfcsvfile), index= False)
    print("\n--- Results have been saved ---")
    print("\nThat took ", (time.monotonic()-starttime)/60, " minutes")
#model_to_submission_file(model=rf_clf,
#                         full_X_train=prep_train_text_features,
#                         full_y_train=y_enc_labels,
#                         full_X_test=prep_submission_text_features,
#                         nameOfcsvfile="randomforest_model.csv")
##0.75663
#model_to_submission_file(model=lgr_clf,
#                         full_X_train=prep_train_text_features,
#                         full_y_train=y_enc_labels,
#                         full_X_test=prep_submission_text_features,
#                         nameOfcsvfile="logistic_regression_model.csv")
##0.78751
ovr_svm = OneVsRestClassifier(svm_clf, n_jobs = 4)

model_to_submission_file(model=ovr_svm,
                         full_X_train=prep_train_text_features,
                         full_y_train=y_enc_labels,
                         full_X_test=prep_submission_text_features,
                         nameOfcsvfile="svm_model.csv")
#0.81999
#ovr_svm = OneVsRestClassifier(p_svm_clf, n_jobs = 4)
#vt_clf = VotingClassifier(estimators=[('rf',rf_clf), 
#                                      ('log_reg',lgr_clf),
#                                      ('SVM+ovr', ovr_svm)],
#                                      voting = 'soft',
#                                      weights = [1,2,4])
#vt_clf.fit(prep_train_text_features,y_enc_labels)
#
#submission_pred = vt_clf.predict(prep_submission_text_features)
#test_cuisine = lb_enc.inverse_transform(submission_pred)
#
#test_id = [recipe['id'] for recipe in test]
#
#submission_df = pd.DataFrame({'id':test_id, 'cuisine':test_cuisine},columns = ['id','cuisine'])
#submission_df.to_csv('votingclassifier.csv', index= False)
##0.81989
#submission score
model_testing_score = [0.75663, 0.78751, 0.81999, 0.81989]

pd.DataFrame({"Model":model_names,
              "Training Score":best_model_training_score,
              "Testing Score": best_model_testing_score,
              "Submission Score": model_testing_score},
             columns=["Model","Training Score", "Testing Score", "Submission Score"])

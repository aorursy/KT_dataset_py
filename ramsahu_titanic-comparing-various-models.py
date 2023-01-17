import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from flashtext import KeywordProcessor
from sklearn.impute import KNNImputer

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_validate,ShuffleSplit
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv("../input/titanic/train.csv")
df_test = pd.read_csv("../input/titanic/test.csv")
scaler = MinMaxScaler()

#Normalising the Titles
dict_title = {
    "Officer" : ["Capt","Col","Major","Dr","Rev",],
    "Royalty": ["Jonkheer","Don","Dona", "Sir","the Countess","Lady",],
    "Mrs": ["Mme","Ms","Mrs",],
    "Mr" : ["Mr"],
    "Miss": ["Miss","Mlle",],
    "Master" : ["Master"],
}

kp_title_norm = KeywordProcessor()
kp_title_norm.add_keywords_from_dict(dict_title)

def pipeline(df_):
    """
    Pipeline for Feature Engineering, Missing value imputation, and Scaling
    """
    
    ## Missing value 
    # Fill missing value of "Embarked" by mode
    df_['Embarked'].fillna(df_['Embarked'].mode()[0], inplace=True)

    ## Creating one hot encoder
    df_["Sex_bin"] = pd.get_dummies(df_["Sex"], prefix=['Sex_bin'], drop_first=True)
    df_temp = pd.get_dummies(df_["Embarked"], prefix='Embarked', drop_first=True)
    df_ = df_.join(df_temp)

    df_temp = pd.get_dummies(df_["Pclass"], prefix='Pclass', drop_first=True)
    df_ = df_.join(df_temp)

    count_Ticket = dict(df_.Ticket.value_counts())
    df_["count_Ticket"] = df_["Ticket"].apply(lambda x: count_Ticket[x])
    
    df_['Title'] = df_['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]
    df_["Title_norm"] = df_["Title"].apply(lambda x: kp_title_norm.extract_keywords(x)[0])

    df_temp = pd.get_dummies(df_["Title_norm"], prefix='Title_norm',)
    df_ = df_.join(df_temp)

    df_['Is_Married'] = 0
    df_['Is_Married'].loc[df_['Title'] == 'Mrs'] = 1

    df_.drop(["Name", 'Sex', "Embarked", "Pclass", "Ticket", "Cabin", "Title", "Title_norm"], axis = 1, inplace = True)

    # Fill missing value of "Embarked" by  KNNimputer
    list_columns_temp = list(df_.columns)
    imputer = KNNImputer(n_neighbors=2)
    df_ = pd.DataFrame(imputer.fit_transform(df_), columns= list_columns_temp)

    df_[['Age', 'Fare']] = scaler.fit_transform(df_[['Age', 'Fare']])
    
    return df_
df_train_ = df_train.copy()
df_test_ = df_test.copy()
df_train__preprocessed = pipeline(df_train_)
df_train__preprocessed.head(2)
del df_train__preprocessed["PassengerId"]
col_df = list(df_train__preprocessed.columns)
col_X = [elem for elem in col_df if elem != "Survived"]
col_Y = ["Survived"]
# create training and testing vars
X_train, X_val, y_train, y_val = train_test_split(df_train__preprocessed[col_X], df_train__preprocessed[col_Y], test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_val.shape, y_val.shape)
## Predicting on the Test Data
def get_predictions(clf_, if_probability=False):
    df= pd.DataFrame()
    df["PassengerId"] = df_test_["PassengerId"]
    df["Survived"] = clf_.predict(df_test__preprocessed[col_X])
    if not if_probability:
        df["Survived"] = df["Survived"].apply(lambda x: int(x))
    else:
        predict_train = clf_.predict(df_test__preprocessed[col_X])
        df["Survived"] = [1 if (i>0.5) else 0 for i in predict_train]
    return df
from sklearn.linear_model import LogisticRegression

# Initialize the model
LogReg = LogisticRegression(max_iter=60)

# fit the model with the training data
LogReg.fit(X_train,y_train)

# # coefficeints of the trained model
# print('Coefficient of model :', LogReg.coef_)

# # intercept of the model
# print('Intercept of model',LogReg.intercept_)

# predict the target on the train dataset
predict_train = LogReg.predict(X_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = LogReg.predict(X_val)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
df_test__preprocessed = pipeline(df_test_)
df_predict_test_LogReg = get_predictions(LogReg)
path_LogReg_predict = "LogReg_predict_17_june.csv"
df_predict_test_LogReg.to_csv(path_LogReg_predict, index=False)
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree classifer object
DecisionTree_clf = DecisionTreeClassifier(criterion="entropy",)

# Train Decision Tree Classifer
DecisionTree_clf.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = DecisionTree_clf.predict(X_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = DecisionTree_clf.predict(X_val)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
# We can clearly see that we have overfitted our model, Hence, we will tune it to get better results.

# ## Predicting on the Test Data
df_predict_test_DecisionTree = get_predictions(DecisionTree_clf)

path_DecisionTree_predict = "DecisionTree_predict_17_june.csv"
df_predict_test_DecisionTree.to_csv(path_DecisionTree_predict, index=False)
# Necessary imports 
from scipy.stats import randint 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import RandomizedSearchCV 
  
# Creating the hyperparameter grid  
param_dist = {"max_depth": [3, 4,6, None], 
              "max_features": randint(1, 17), 
              "min_samples_leaf": randint(1, 5), 
              "criterion": ["gini", "entropy"]} 
  
# Instantiating Decision Tree classifier 
DecisionTree_clf = DecisionTreeClassifier() 
  
# Instantiating RandomizedSearchCV object 
DecisionTree_clf_cv = RandomizedSearchCV(DecisionTree_clf, param_dist, cv = 5) 

# Train Decision Tree Classifer
DecisionTree_clf_cv.fit(X_train,y_train)

# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(DecisionTree_clf_cv.best_params_)) 
print("Best score is {}".format(DecisionTree_clf_cv.best_score_)) 

# predict the target on the train dataset
predict_train = DecisionTree_clf_cv.predict(X_train)
# print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = DecisionTree_clf_cv.predict(X_val)
# print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
# Create Decision Tree classifer object

best_criterion_ = DecisionTree_clf_cv.best_params_["criterion"]
best_max_depth_ = DecisionTree_clf_cv.best_params_["max_depth"]
best_max_features_ = DecisionTree_clf_cv.best_params_["max_features"]
best_min_samples_leaf_ = DecisionTree_clf_cv.best_params_["min_samples_leaf"]

DecisionTree_clf_tuned = DecisionTreeClassifier(criterion=best_criterion_, max_depth= best_max_depth_, max_features= best_max_features_,
                                                min_samples_leaf= best_min_samples_leaf_)

# Train Decision Tree Classifer
DecisionTree_clf_tuned.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = DecisionTree_clf_tuned.predict(X_train)
# print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = DecisionTree_clf_tuned.predict(X_val)
# print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
df_predict_test_DecisionTree_tuned = get_predictions(DecisionTree_clf_tuned)
path_DecisionTree_Tuned_predict = "DecisionTree_Tuned_predict_17_june.csv"
df_predict_test_DecisionTree_tuned.to_csv(path_DecisionTree_Tuned_predict, index=False)
#Import Library
from sklearn.ensemble import RandomForestClassifier

# Create Linear RandomForestClassifier object
random_forest_classifier = RandomForestClassifier(n_estimators=100)

# Train Classifer
random_forest_classifier.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = random_forest_classifier.predict(X_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = random_forest_classifier.predict(X_val)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
#Import Library
from sklearn import svm

# Create Linear SVM object
SVM_classifier = svm.LinearSVC(random_state=20)

# Train Decision Tree Classifer
SVM_classifier.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = SVM_classifier.predict(X_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = SVM_classifier.predict(X_val)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
df_predict_test_SVM_classifier = get_predictions(SVM_classifier)
# df_predict_test_SVM_classifier.head(2)

path_SVM_classifier_predict = "SVM_classifier_predict_17_june.csv"
df_predict_test_SVM_classifier.to_csv(path_SVM_classifier_predict, index=False)
import lightgbm as lgb

# Create Light GB object
LGB_classifier = lgb.LGBMClassifier()

# fit the model with the training data
LGB_classifier.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = LGB_classifier.predict(X_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = LGB_classifier.predict(X_val)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
df_predict_test_LGB_classifier = get_predictions(LGB_classifier)

path_LGB_classifier_predict = "LGB_classifier_predict_17_june.csv"
df_predict_test_LGB_classifier.to_csv(path_LGB_classifier_predict, index=False)
from xgboost import XGBClassifier

# fit model on training data
XGB_classifier = XGBClassifier()

# fit the model with the training data
XGB_classifier.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = XGB_classifier.predict(X_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = XGB_classifier.predict(X_val)

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
df_predict_test_XGB_classifier = get_predictions(XGB_classifier)
path_XGB_classifier_predict = "XGB_classifier_predict_17_june.csv"
df_predict_test_XGB_classifier.to_csv(path_XGB_classifier_predict, index=False)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

## Hyper Parameter Optimization
params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}
classifier=XGBClassifier()
XGB_clf_cv=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc', cv=5)
XGB_clf_cv.fit(X_train,y_train)


# Print the tuned parameters and score 
print("Tuned Decision Tree Parameters: {}".format(XGB_clf_cv.best_params_)) 
print("Best score is {}".format(XGB_clf_cv.best_score_)) 

# predict the target on the train dataset
predict_train = XGB_clf_cv.predict(X_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = XGB_clf_cv.predict(X_val)
# print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
# Create Decision Tree classifer object
czzcc
best_criterion_ = DecisionTree_clf_cv.best_params_["criterion"]
best_max_depth_ = DecisionTree_clf_cv.best_params_["max_depth"]
best_max_features_ = DecisionTree_clf_cv.best_params_["max_features"]
best_min_samples_leaf_ = DecisionTree_clf_cv.best_params_["min_samples_leaf"]

DecisionTree_clf_tuned = DecisionTreeClassifier(criterion=best_criterion_, max_depth= best_max_depth_, max_features= best_max_features_,
                                                min_samples_leaf= best_min_samples_leaf_)

# Train Decision Tree Classifer
DecisionTree_clf_tuned.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = DecisionTree_clf_tuned.predict(X_train)
# print('Target on train data',predict_train) 

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = DecisionTree_clf_tuned.predict(X_val)
# print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
import tensorflow as tf
## Single Hidden layer

ann_classifier = tf.keras.models.Sequential()
ann_classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
# ann_classifier.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann_classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann_classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann_classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)


# # fit the model with the training data
# XGB_classifier.fit(X_train,y_train)

# predict the target on the train dataset
predict_train = ann_classifier.predict(X_train)
predict_train = [1 if (i>0.5) else 0 for i in predict_train ]

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = ann_classifier.predict(X_val)
predict_test = [1 if (i>0.5) else 0 for i in predict_test ]

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
df_predict_test_ANN_classifier = get_predictions(ann_classifier, if_probability=True)
path_ANN_classifier_predict = "ANN_classifier_predict_17_june.csv"
df_predict_test_ANN_classifier.to_csv(path_ANN_classifier_predict, index=False)
## Multi-Hidden layer

ann_classifier_2 = tf.keras.models.Sequential()
ann_classifier_2.add(tf.keras.layers.Dense(units=12, activation='relu'))
ann_classifier_2.add(tf.keras.layers.Dense(units=8, activation='relu'))
ann_classifier_2.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann_classifier_2.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann_classifier_2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann_classifier_2.fit(X_train, y_train, batch_size = 32, epochs = 100)


# predict the target on the train dataset
predict_train = ann_classifier_2.predict(X_train)
predict_train = [1 if (i>0.5) else 0 for i in predict_train ]

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_train,predict_train)
print('accuracy_score on train dataset : ', accuracy_train)

# predict the target on the test dataset
predict_test = ann_classifier_2.predict(X_val)
predict_test = [1 if (i>0.5) else 0 for i in predict_test ]

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_val,predict_test)
print('accuracy_score on Validation dataset : ', accuracy_test)
df_predict_test_ANN_classifier_2 = get_predictions(ann_classifier_2, if_probability=True)
path_ANN_classifier_2_predict = "ANN_classifier_2_predict_17_june.csv"
df_predict_test_ANN_classifier_2.to_csv(path_ANN_classifier_2_predict, index=False)
# from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process, neural_network
from sklearn import ensemble

vote_est = [
    ("LogReg", LogisticRegression(max_iter=60)),
    ("DecisionTree_clf_tuned", DecisionTreeClassifier(criterion=best_criterion_, max_depth= best_max_depth_, max_features= best_max_features_,
                                                min_samples_leaf= best_min_samples_leaf_)),
    ("SVC", svm.SVC(max_iter = 500000,probability=True,kernel='linear',C=0.025)),
    ("LGB_classifier", lgb.LGBMClassifier()),
    ("XGB_classifier", XGBClassifier()),]
#Hard Vote
cv_split = ShuffleSplit(n_splits = 10, test_size = .3, train_size = .7, random_state = 0 ) 
vote_hard = ensemble.VotingClassifier(estimators = vote_est , voting = 'hard')

vote_hard_cv = cross_validate(vote_hard, df_train__preprocessed[col_X], 
                              df_train__preprocessed[col_Y], cv  = cv_split, return_train_score=True)

vote_hard.fit(df_train__preprocessed[col_X], df_train__preprocessed[col_Y])

print("Hard Voting Training w/bin score mean: {:.2f}". format(vote_hard_cv['train_score'].mean()*100)) 
print("Hard Voting Test w/bin score mean: {:.2f}". format(vote_hard_cv['test_score'].mean()*100))
print("Hard Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_hard_cv['test_score'].std()*100*3))
print('-'*10)
df_predict_test_Ensemble_hard_vote_classifier = get_predictions(vote_hard)
path_Ensemble_hard_vote_classifier_predict = "Ensemble_hard_vote_classifier_predict_17_june.csv"
df_predict_test_Ensemble_hard_vote_classifier.to_csv(path_Ensemble_hard_vote_classifier_predict, index=False)
#Soft Vote
vote_soft = ensemble.VotingClassifier(estimators = vote_est , voting = 'soft')

vote_soft_cv = cross_validate(vote_soft, df_train__preprocessed[col_X], 
                              df_train__preprocessed[col_Y], cv=cv_split, return_train_score=True)

vote_soft.fit(df_train__preprocessed[col_X], df_train__preprocessed[col_Y])

print("Soft Voting Training w/bin score mean: {:.2f}". format(vote_soft_cv['train_score'].mean()*100)) 
print("Soft Voting Test w/bin score mean: {:.2f}". format(vote_soft_cv['test_score'].mean()*100))
print("Soft Voting Test w/bin score 3*std: +/- {:.2f}". format(vote_soft_cv['test_score'].std()*100*3))
print('-'*10)
df_predict_test_Ensemble_soft_vote_classifier = get_predictions(vote_soft)
path_Ensemble_soft_vote_classifier_predict = "Ensemble_soft_vote_classifier_predict_17_june.csv"
df_predict_test_Ensemble_soft_vote_classifier.to_csv(path_Ensemble_soft_vote_classifier_predict, index=False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")

data.head()
test = pd.read_csv("../input/test.csv")

test.head()
%%time

print("**"*45)

print('The size of our train and test datasets are: \nTrain : {}\nTest: {}'.format(data.shape, test.shape))

print("**"*45)
data.describe()
data.dtypes.value_counts()
test.dtypes.value_counts()
# We check for any missing or nan values in the data and test sets

null_features = data[data.columns[data.isnull().any()]].sum().sort_values()

missing_train = pd.DataFrame({'Null' : null_features})

missing_train

nulltest_features = test[test.columns[test.isnull().any()]].sum().sort_values()

missing_test = pd.DataFrame({'Null' : nulltest_features})

missing_test

#  :) :) :) Train and test sets have not missing values. GREEEEEEEEEAAAAATTTTT!!!!! :) :) :)
data['target'].value_counts(normalize = True)
sns.set(style = 'darkgrid')

ax = sns.countplot(x = 'target', data = data)
#we check for skewness in  data



skew_limit = 0.75

skew_vals = data.skew()



skew_cols = (skew_vals

             .sort_values(ascending=False)

             .to_frame()

             .rename(columns={0:'Skewness'})

            .query('abs(Skewness) > {0}'.format(skew_limit)))



skew_cols

# Correlation between the features and the predictor- SalePrice

predictor = data['target']

features = [x for x in data.columns if x != 'target']

correlations = data[features].corrwith(predictor)

correlations = correlations.sort_values(ascending = False)



# correlations

corrs = (correlations

            .to_frame()

            .reset_index()

            .rename(columns={'level_0':'feature1',

                                0:'Correlations'}))



corrs.head()
# Get the absolute values for sorting

corrs['Abs_correlation'] = corrs.Correlations.abs()

corrs.head()
# Most correlated features wrt the abs_correlations

corrs.sort_values('Correlations', ascending = False).query('Abs_correlation>0.45')

#Importing my classifiers and scoring metrics

from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from sklearn.metrics import classification_report

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
# We split the data into train and validation set

# We are not going to need the id_code column nor the target column  in X

X = data.drop(columns = ['target', 'ID_code'])

y = data['target']



X_train, X_val, y_train, y_val = train_test_split(X, y,

                                                  test_size = 0.3,

                                                  random_state=42)

print('Training set shape:')

print('X_train.shape:{}\t y_train.shape: {}'.format(X_train.shape, y_train.shape))

print('\nValidation set shape:')

print('X_val.shape:{} \t y_val.shape: {}'.format(X_val.shape, y_val.shape))
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1,

                                      class_weight= None, max_features = 0.5, 

                                       min_samples_leaf = 100)
#Train the model

random_forest.fit(X_train, y_train)
pred_rf = random_forest.predict(X_val)

print ('accuracy_score for Random_forest with unbalanced classses')



accuracy = accuracy_score(y_val, pred_rf)

print("Accuracy: {}".format (accuracy))

#Getting the accuracy_score and ta roc_score for the random_forest

print ('\nroc_auc_score for Random_forest with unbalanced classes')

pred = random_forest.predict_proba(X_val)[:,1]

roc_auc = roc_auc_score(y_val, pred)

print(roc_auc)
data['target'].value_counts(normalize = True)
#submission 1
test_all = test.drop('ID_code', axis = 1)

test_all.head()
predict_1 = random_forest.predict(test_all)

#random_forest.predict_proba(X_val)[:,1]





solution_1 = pd.DataFrame({'ID_code': test['ID_code'], "target" : predict_1})



# #creating csv file



#solution_1.to_csv("santander_1.csv", index = False)

solution_1.head()
#only predicts the majority class 0 

np.count_nonzero(solution_1['target']==0)
rand_f1 = random_forest.fit(X_train[['var_5']], y_train)

accuracy_one = accuracy_score(y_val,  rand_f1.predict(X_val[['var_5']]))

print("Accuracy_one: {}".format (accuracy_one))



roc_predict = rand_f1.predict_proba(X_val[['var_5']])

roc_auc_score_one = roc_auc_score(y_val, (roc_predict)[:,1])



print("roc_one: {}".format (roc_auc_score_one))
rf = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1,

                                      class_weight= 'balanced', max_features = 0.5, 

                                       min_samples_leaf = 100)
rf.fit(X_train, y_train)
predict_balanced = rf.predict(X_val)

print ('accuracy_score for Random_forest with balanced classes')

print (accuracy_score(y_val, rf.predict(X_val)))



print ('\nroc_auc_score for Random_forest with class_weight = balanced classes')



roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:,1])

print(roc_auc)
#Submission 2 (scores = 0.788 on public leaderboard ~79%)

prediction_2 = rf.predict_proba(test_all)[:,1]

solution_2 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_2 })



#creating csv file



#solution_2.to_csv("santander_2.csv", index = False)

solution_2.shape

  
#Using SMOTE for class imbalance in target

from imblearn.over_sampling import SMOTE

from collections import Counter

sm = SMOTE(random_state=42)

X_resampled, y_resampled = sm.fit_resample(X, y)

print('Resampled dataset shape %s' % Counter(y_resampled))
# Now we split the resmpled train set into train and validation sets

x_train_res, x_val_res, y_train_res, y_val_res = train_test_split(X_resampled,

                                                    y_resampled,

                                                    test_size = 0.3,

                                                    random_state=2019)
#Train the model with the same random forest algorithm

rf_1 = random_forest.fit(x_train_res, y_train_res)
#Getting the f1_score and ta classifiaton report for the random_forest



# target_names = ['class 0', 'class 1']



pred_1 = rf_1.predict(x_val_res)

print ('accuracy_score for Random_forest with "smote" oversampling before splitting')

accuracy_smote_1 = accuracy_score(y_val_res, pred_1)

print (accuracy_smote_1)



print ('\nroc_auc_score for Random_forest with "smote" oversampling before splitting')

roc_auc_smote_1 = roc_auc_score(y_val_res, rf_1.predict_proba(x_val_res)[:,1])

print(roc_auc_smote_1) 



# print ('\nClassification report for Random_forest with "smote" oversampling before splitting')

# print(classification_report(y_val_res,pred_1, target_names=target_names))
#scores 0.662 on public leaderboard

prediction_3 = rf_1.predict_proba(test_all)[:,1]
solution_3 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_3 })

#solution_3 = pd.DataFrame({'ID_code': test, "target" : pred_1 })



#creating csv file



#solution_3.to_csv("santander_3.csv", index = False)

#solution_3.sample(10)
# #lets get the accuracy and roc_auc_score for the test data too.

print('The accuracy_score for the test data is:')

accuracy_test_1 = accuracy_score(data['target'], rf_1.predict(test_all))

print(accuracy_test_1)



print('The roc_auc_score for the test data is:')                               

roc_auc_test_1 = roc_auc_score(data['target'], prediction_3)

print(roc_auc_test_1) 



#solution_3 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_3 })

#solution_3 = pd.DataFrame({'ID_code': test, "target" : pred_1 })





  
#train the model with the same random forest algorithm

x_tres, y_tres = sm.fit_sample(X_train, y_train)
#fit the random forest classifier on the split train sets

rf_2 = random_forest.fit(x_tres, y_tres)
#Getting the f1_score and ta classifiaton report for the random_forest



# target_names = ['class 0', 'class 1']

pred_2 = rf_2.predict(X_val)

print ('F1_socre for Random_forest with "smote" oversampling after splitting')

accuracy_smote_2 = accuracy_score(y_val, pred_2)

print (accuracy_smote_2)



print ('\nroc_auc_score for Random_forest with "smote" oversampling after splitting')

roc_auc_smote_2 = roc_auc_score(y_val, rf_2.predict_proba(X_val)[:,1])

print(roc_auc_smote_2)  



# print ('\nClassification report for Random_forest with "smote" oversampling after splitting')

# print(classification_report(y_val, pred_2, target_names=target_names))
#score 0.670 on public leaderboard

prediction_4 = rf_2.predict_proba(test_all)[:,1]
#Creating a submission file



solution_4 = pd.DataFrame({'ID_code': test['ID_code'], "target" : prediction_4 })

#solution_4 = pd.DataFrame({'ID_code': test, "target" : pred_2 })



# #creating csv file



# solution_4.to_csv("santander_4.csv", index = False)

# solution_4.sample(10)


print('The accuracy_score for the test data resampled after splitting is:')

accuracy_test_2 = accuracy_score(data['target'], rf_2.predict(test_all))

print(accuracy_test_2)



print('The roc_auc_score for the test data resampled after splitting:')                               

roc_auc_test_2 = roc_auc_score(data['target'], prediction_4)

print(roc_auc_test_2) 



  
#putting it all together



# REsults with Smote on data befoe splitting

print ('Validation Results for smote before splitting')

print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_smote_1, roc_auc_smote_1))



print ('\nTest Results for smote before splitting')

print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_test_1, roc_auc_test_1))

#results with smote on x_train and y_train only



print ('Validation Results for smote after splitting i.e on x_train and y_train')

print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_smote_2, roc_auc_smote_2))



print ('\nTest Results for smote after splitting i.e on x_train and y_train')

print ('accuracy_score: {} \nroc_auc_score : {}'.format(accuracy_test_2, roc_auc_test_2))
print ('Roc_auc_score for scenario 1: applying Smote before splitting:')

print ('roc_auc_score for validation set: {} \nroc_auc_score for test : {}'.format(roc_auc_smote_1, roc_auc_test_1))

print("\n")

print ('Roc_auc_score for scenario 2: applying Smote after splitting(on X_train and y_train):')

print ('roc_auc_score for validation set: {} \nroc_auc_score for test : {}'.format(roc_auc_smote_2, roc_auc_test_2))
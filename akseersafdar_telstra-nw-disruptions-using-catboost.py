# making the initial imports



import numpy as np

import pandas as pd

import os

import seaborn as sns

sns.set_style("whitegrid")

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



#print the directory items

print(os.listdir('../input/telstra-recruiting-network'))

#reading the files



train = pd.read_csv('../input/telstra-recruiting-network/train.csv')

test = pd.read_csv('../input/telstra-recruiting-network/test.csv')

severity_type = pd.read_csv('../input/telstra-recruiting-network/severity_type.csv', error_bad_lines= False, warn_bad_lines= False)

resource_type = pd.read_csv('../input/telstra-recruiting-network/resource_type.csv', error_bad_lines= False, warn_bad_lines= False)

log_failure = pd.read_csv('../input/telstra-recruiting-network/log_feature.csv', error_bad_lines= False, warn_bad_lines= False)

event_type = pd.read_csv('../input/telstra-recruiting-network/event_type.csv', error_bad_lines=False, warn_bad_lines= False)
#printing the shape of all given files



print('The shape of test set is: {}\n'.format(test.shape))

print('The shape of train set is: {}\n'.format(train.shape))

print('The shape of severity_type is: {}\n'.format(severity_type.shape))

print('The shape of resource_type is: {}\n'.format(resource_type.shape))

print('The shape of log_failure is: {}\n'.format(log_failure.shape))

print('The shape of event_type is: {}'.format(event_type.shape))
#id column in event_types is an object



event_type.dtypes
#convert the id column to numeric data type



event_type['id']=pd.to_numeric(event_type['id'],errors='coerce')
#checking the shape of training set



train.shape
#checking the head of training file before merging it with other files



train.head()
#merging the data sets to have all the available info



train_1 = train.merge(severity_type, how = 'left', left_on='id', right_on='id')

train_2 = train_1.merge(resource_type, how = 'left', left_on='id', right_on='id')

train_3 = train_2.merge(log_failure, how = 'left', left_on='id', right_on='id')

train_4 = train_3.merge(event_type, how = 'left', left_on='id', right_on='id')
#checking the head after merging



train_4.head()
#checking the nulls in each column



train_4.isnull().sum()
#do the head method on training file.

train_4.head(20)
#dropping the duplicate records



train_4.drop_duplicates(subset= 'id', keep= 'first', inplace = True)
#checking the shape of training file after dropping duplicate records



train_4.shape
#count plot for fault severity



plt.figure(figsize = (8,6))

sns.countplot(train_4['fault_severity'])

plt.show()
#count plot for severity type



plt.figure(figsize = (8,6))

sns.countplot(train_4['severity_type'])

plt.show()
#count plot for resource type



plt.figure(figsize = (14,6))

sns.countplot(train_4['resource_type'])

plt.tight_layout()

plt.show()
#plotting the correlation matrix



plt.figure(figsize = (8,6))

sns.heatmap(train_4.corr(), vmax = 0.8, linewidths= 0.01, square= True, 

           annot= True, cmap= 'viridis', linecolor= 'white')



plt.title('Correlation Matrix', fontsize = 15)

plt.show()
#importing the catboost and train test split



from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split
#splitting into X and y (training data and training labels)



X = train_4[['id', 'location', 'severity_type', 'resource_type',

       'log_feature', 'volume', 'event_type']]

y = train_4.fault_severity
#divide the training set into train/validation set with 20% set aside for validation. 



from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=101)
#as we know that we can give categorical features to catboost to make best use of its performance. 





categorical_features_indices = np.where(X_train.dtypes == object)[0]
#using pool to make the training and validation sets



train_dataset = Pool(data=X_train,

                     label=y_train,

                     cat_features=categorical_features_indices)



eval_dataset = Pool(data=X_validation,

                    label=y_validation,

                    cat_features=categorical_features_indices)

# Initialize CatBoostClassifier



model = CatBoostClassifier(iterations=1000,

                           learning_rate=1,

                           depth=2,

                           loss_function='MultiClass',

                           random_seed=1,

                           bagging_temperature=22,

                           od_type='Iter',

                           metric_period=100,

                           od_wait=100)
# Fit model



model.fit(train_dataset, eval_set= eval_dataset, plot= True)
# Get predicted classes



preds_class = model.predict(eval_dataset)
# Get predicted probabilities for each class



preds_proba = model.predict_proba(eval_dataset)
#we are getting the probabilities in this format.



preds_proba
#checking the head



test.head()
#checking the shape of test set before merging with other files. 



test.shape
#merging the data sets to combine all the needed info



test_1 = test.merge(severity_type, how = 'left', left_on='id', right_on='id')

test_2 = test_1.merge(resource_type, how = 'left', left_on='id', right_on='id')

test_3 = test_2.merge(log_failure, how = 'left', left_on='id', right_on='id')

test_4 = test_3.merge(event_type, how = 'left', left_on='id', right_on='id')
#checkingk the head 20 records



test_4.head(20)
#removing the duplicates.



test_4.drop_duplicates(subset= 'id', keep= 'first', inplace = True)
#checkingk the shape of test set again



test_4.shape
#checking for any null values. 



test_4.isnull().sum()
#making predictions on test set



predict_test=model.predict_proba(test_4)

pred_df=pd.DataFrame(predict_test,columns=['predict_0', 'predict_1', 'predict_2'])

submission_cat=pd.concat([test[['id']],pred_df],axis=1)

submission_cat.to_csv('sub_cat_1.csv',index=False,header=True)
#having a look at the submission file



submission_cat.head()
#making the imports



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
#checking the head of training file



train_4.head()
#initialize the label encoder



lb = LabelEncoder()
#apply the label encoder to all the categorical columns



train_4['location'] = lb.fit_transform(train_4['location'])

train_4['severity_type'] = lb.fit_transform(train_4['severity_type'])

train_4['resource_type'] = lb.fit_transform(train_4['resource_type'])

train_4['log_feature'] = lb.fit_transform(train_4['log_feature'])

train_4['event_type'] = lb.fit_transform(train_4['event_type'])
#checking the head of encoded training set



train_4.head(20)
#divide the data into X and y



y = train_4['fault_severity']

X = train_4.drop('fault_severity', axis = 1)
#making training and test sets with 75% and 25% ratio



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
#instantiate and train the model



rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
#making predictions on test set



pred = rfc.predict(X_test)
#confusion matrix and classification report



print('Confusion matrix \n')

print(metrics.confusion_matrix(y_test,pred))

print('*'*80)

print('\n')

print('Classification report \n')

print(metrics.classification_report(y_test,pred))
#making the needed imports



from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
#divide the data into 5 folds



folds = KFold(n_splits= 5, shuffle= True, random_state= 101)
#give the range for parameters to check for grid search



params = {'max_depth':[3,5,7,9],

         'n_estimators':[500,800,1100,1400],

         'min_samples_leaf': [150, 200, 250, 300], 

         'min_samples_split': [300, 350, 400, 450]}



rf = RandomForestClassifier()
#define the model with gridsearch parameters



rf_fin = GridSearchCV(estimator= rf, cv = folds, param_grid= params, scoring= 'accuracy', return_train_score= True)

#traint the model (will take some time)



rf_fin.fit(X_train,y_train)
#getting the results in a dataframe



scores = rf_fin.cv_results_



scores = pd.DataFrame(scores)



scores.head()
#getting the best score



print('The best score was achieved using the parameters: {}'.format(rf_fin.best_params_))
#so now we will use the above parameters to build the model



random_final = RandomForestClassifier(max_depth= 3,

                                      min_samples_leaf= 150,

                                      min_samples_split= 300,

                                      n_estimators= 500)
#fit the model and get the predictions



random_final.fit(X_train,y_train)



pred_fin = random_final.predict(X_test)
#printing the confusion matrix and classification reports



print('Confusion matrix \n')

print(metrics.confusion_matrix(y_test,pred_fin))

print('*'*80)

print('\n')

print('Classification report \n')

print(metrics.classification_report(y_test,pred_fin))
#checking the orignal form of test_4 (without label encoding the data)



test_4.head()
#applying the label encode to categorical columns



test_4['location'] = lb.fit_transform(test_4['location'])

test_4['severity_type'] = lb.fit_transform(test_4['severity_type'])

test_4['resource_type'] = lb.fit_transform(test_4['resource_type'])

test_4['log_feature'] = lb.fit_transform(test_4['log_feature'])

test_4['event_type'] = lb.fit_transform(test_4['event_type'])
#label encoded test set



test_4.head()
# we will use the predict_proba as this is needed format for kaggle submission. 



pred_fin = rfc.predict_proba(test_4)
#making the submission file ready



pred_df=pd.DataFrame(pred_fin,columns=['predict_0', 'predict_1', 'predict_2'])

submission_rf=pd.concat([test[['id']],pred_df],axis=1)

submission_rf.to_csv('sub_random_forest.csv',index=False,header=True)
#checking the submission file. 



submission_rf.head()
import pandas as pd
train_data = pd.read_csv('../input/train_LZdllcl.csv')
test_data = pd.read_csv('../input/test_2umaH9m.csv')
train_data.head()
missing_train_dataset=train_data.copy()
# missing_data_train
new_labels=[]
for data_lab in missing_train_dataset:
    if missing_train_dataset[data_lab].dtypes==object:
        new_labels.append(data_lab)
        
print(new_labels)
missing_train_dataset.drop(labels=new_labels, inplace=True, axis=1)


# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train_set = my_imputer.fit_transform(missing_train_dataset)
imputed_X_train_set


# In[13]:
imputed_X_train_df=pd.DataFrame(imputed_X_train_set, columns=['employee_id', 'no_of_trainings', 'age', 'previous_year_rating',
       'length_of_service', 'KPIs_met >80%', 'awards_won?',
       'avg_training_score', 'is_promoted'])


imputed_X_categorical_train_df = train_data.select_dtypes(exclude=['int64','float64']).copy()
imputed_X_categorical_train_df.fillna(value='Bachelor\'s', inplace=True)

one_hot_encoded_training_predictors_with_categorical=pd.get_dummies(imputed_X_categorical_train_df)

merged_df_predictors=pd.concat([one_hot_encoded_training_predictors_with_categorical,imputed_X_train_df],axis=1)

final_train_predictor=merged_df_predictors.iloc[:,:59]
final_train_target=merged_df_predictors.is_promoted
final_train_predictor.drop(labels=['employee_id'], axis=1, inplace=True)

#final_train_predictor.isnull().sum()
final_train_target.isnull().sum()
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(final_train_predictor,final_train_target,test_size=0.25,random_state=5)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# from sklearn.svm import SVC

# #from sklearn.svm import score 
# model_SVC = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0) #select the algorithm
# model_SVC.fit(X_train,Y_train) # we train the algorithm with the training data and the training output
# prediction_SVC=model_SVC.predict(X_test)
# from sklearn.metrics import confusion_matrix
# cm_SVC = confusion_matrix(Y_test,prediction_SVC.round())
# cm_SVC
# #prediction_SVC
# from sklearn.metrics import classification_report
# target_names = ['class 0', 'class 1']
# print(classification_report(Y_test, prediction_SVC.round(), target_names=target_names))
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(learning_rate=0.01,num_boost_round = 2000,colsample_bytree=1,eta=0.3,eval_metric='mae',max_depth=11,min_child_weight=11,subsample=0.9)
xgb_clf = xgb_clf.fit(X_train,Y_train)
prediction_xgb=xgb_clf.predict(X_test) #now we pass the testing data to the trained algorithm
from sklearn.metrics import confusion_matrix
cm_xgb = confusion_matrix(Y_test,prediction_xgb)
# cm_xgb
# from sklearn.metrics import classification_report
# target_names = ['class 0', 'class 1']
# print(classification_report(Y_test, prediction_xgb, target_names=target_names))
# from keras import models
# from keras import layers
# from keras.layers import Dropout
# model = models.Sequential()
# model.add(layers.Dense(100,activation='relu',input_shape=(58,)))
# model.add(layers.Dense(100,activation='relu'))
# model.add(Dropout(0.25))
# model.add(layers.Dense(100,activation='relu'))
# model.add(Dropout(0.25))
# model.add(layers.Dense(100,activation='relu'))
# model.add(Dropout(0.50))
# model.add(layers.Dense(1,activation='sigmoid'))
# from keras import optimizers
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])

# model.fit(X_train,Y_train,epochs=50,batch_size=512)
# results = model.predict_classes(X_test)
# from sklearn.metrics import confusion_matrix
# cm_keras = confusion_matrix(Y_test,results)
# cm_keras
# from sklearn.metrics import classification_report
# target_names = ['class 0', 'class 1']
# print(classification_report(Y_test, results, target_names=target_names))
test_data.head()
missing_test_dataset=test_data.copy()

test_labels=[]
for data_lab in missing_test_dataset:
    if missing_test_dataset[data_lab].dtypes==object:
        test_labels.append(data_lab)
        
print(new_labels)
missing_test_dataset.drop(labels=new_labels, inplace=True, axis=1)


# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train_set = my_imputer.fit_transform(missing_test_dataset)
imputed_X_train_set


# In[13]:
imputed_Y_test_df=pd.DataFrame(imputed_X_train_set, columns=['employee_id', 'no_of_trainings', 'age', 'previous_year_rating',
       'length_of_service', 'KPIs_met >80%', 'awards_won?',
       'avg_training_score'])


imputed_Y_categorical_test_df = test_data.select_dtypes(exclude=['int64','float64']).copy()
imputed_Y_categorical_test_df.fillna(value='Bachelor\'s', inplace=True)

one_hot_encoded_testing_predictors_with_categorical=pd.get_dummies(imputed_Y_categorical_test_df)

merged_df_predictors_test=pd.concat([one_hot_encoded_testing_predictors_with_categorical,imputed_Y_test_df],axis=1)

final_test_predictor=merged_df_predictors_test.iloc[:,:59]
final_test_predictor.drop(labels=['employee_id'], axis=1, inplace=True)

final_test_predictor.isnull().sum()
# final_test_target.isnull().sum()
# from keras import models
# from keras import layers
# from keras.layers import Dropout
# model = models.Sequential()
# model.add(layers.Dense(100,activation='relu',input_shape=(58,)))
# model.add(layers.Dense(100,activation='relu'))
# model.add(Dropout(0.25))
# model.add(layers.Dense(100,activation='relu'))
# model.add(Dropout(0.25))
# model.add(layers.Dense(100,activation='relu'))
# model.add(Dropout(0.50))
# model.add(layers.Dense(1,activation='sigmoid'))
# from keras import optimizers
# from keras import metrics
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='binary_crossentropy',metrics=[metrics.binary_accuracy])

# model.fit(final_train_predictor,final_train_target,epochs=50,batch_size=512)
# results = model.predict_classes(final_test_predictor)
# results_df=pd.DataFrame(results,columns=['predicted'])
# len(results_df.predicted)
submission_df=pd.read_csv('../input/sample_submission_M0L0uXE.csv')
# get_final_prediction=pd.concat([submission_df,results_df], axis=1)
# get_final_prediction.drop(labels=['is_promoted'], inplace=True, axis=1)
# get_final_prediction.rename(columns={'predicted':'is_promoted'}, inplace=True)
# get_final_prediction
# get_final_prediction.to_csv('First_Submission_1.csv',  index=False)
# import xgboost as xgb

# xgb_clf = xgb.XGBClassifier(learning_rate=0.01,num_boost_round = 2000,colsample_bytree=1,eta=0.3,eval_metric='mae',max_depth=11,min_child_weight=11,subsample=0.9)
# xgb_clf = xgb_clf.fit(final_train_predictor,final_train_target)
final_test_predictor = sc_X.transform(final_test_predictor)
prediction_xgb=xgb_clf.predict(final_test_predictor) #now we pass the testing data to the trained algorithm

from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test,prediction)
'''from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=xgb_clf,X=X_train,y=y_train,cv=10)
print('Cross validation accuracies is',accuracies.mean())'''
results_df_xgb=pd.DataFrame(prediction_xgb,columns=['predicted'])
get_final_prediction_xgb=pd.concat([submission_df,results_df_xgb], axis=1)
get_final_prediction_xgb.drop(labels=['is_promoted'], inplace=True, axis=1)
get_final_prediction_xgb.rename(columns={'predicted':'is_promoted'}, inplace=True)
# you could use any filename. We choose submission here
get_final_prediction_xgb.to_csv('Xgboost_Scaled.csv', index=False)
'''
print('The accuracy of the xgb classifier is {:.2f} out of 1 on training data'.format(xgb_clf.score(X_train, y_train)))
print('The accuracy of the xgb classifier is {:.2f} out of 1 on test data'.format(xgb_clf.score(X_test, y_test)))
'''
# get_final_prediction_xgb.is_promoted.unique()
# from sklearn.svm import SVC
# final_test_predictor = sc_X.transform(final_test_predictor)

# #from sklearn.svm import score 
# #model_SVC = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0) #select the algorithm
# #model_SVC.fit(X_train,Y_train) # we train the algorithm with the training data and the training output
# prediction_SVC_validation=model_SVC.predict(final_test_predictor)
# results_df_SVC=pd.DataFrame(prediction_SVC_validation,columns=['predicted'])
# get_final_prediction_SVC=pd.concat([submission_df,results_df_SVC], axis=1)
# get_final_prediction_SVC.drop(labels=['is_promoted'], inplace=True, axis=1)
# get_final_prediction_SVC.rename(columns={'predicted':'is_promoted'}, inplace=True)
# get_final_prediction_SVC.to_csv('First_Submission_Kernel.csv',  index=False)

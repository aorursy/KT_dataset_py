#Importing Libraries



import numpy as np 

import pandas as pd 



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import MinMaxScaler



from sklearn.feature_selection import SelectPercentile

from sklearn.feature_selection import f_classif 



from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score
#Reading Data in



data = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
data.info()
data.shape
data.head(10)
data.tail(10)
#Renaming columns to have all the headers in lowercase to avoid confusing



data.rename(columns = lambda x: x.strip().lower().replace("-", "_"), inplace=True)
#Checking columns new names



data.head()
#Creating a list contains columns names for faster itering over them



columns_list = data.columns.to_list()

columns_list
#Exploring unique values for each column to understand what we are dealing with



for column in columns_list:

    print(data[column].unique())

    print('')
# Now we'll start manpulating our data to prepare it for further exploring



#Removing unwanted data .. we don't need patientid and appointmentid



data.drop(['patientid', 'appointmentid'], axis = 1, inplace = True)
#Checking modifications



data.head()
#We will drop rows with ages as '-1'



data = data[data.age >= 0]
#Checking modifications



data.age.unique()
#Now we need to deal with date columns 



#First we will convert them to datetime type instead of object



data['scheduledday'] = pd.to_datetime(data['scheduledday']).astype('datetime64[ns]')



data['appointmentday'] = pd.to_datetime(data['appointmentday']).astype('datetime64[ns]')
#Checking modificatoins



data.info()
data.head()
#We notice from tha data that all appointments time at zero hour

#So we will remove time component and deal with date and days only



data['scheduledday'] = data['scheduledday'].dt.date

data['appointmentday'] = data['appointmentday'].dt.date
data.head()
#Now we will create a new column that calculate the waiting time between scheduled and appointment



data['waiting_time_days'] = data['appointmentday'] - data['scheduledday']



data['waiting_time_days'] = data['waiting_time_days'].dt.days
data['waiting_time_days'].unique()
#Trere are some waiting days vaues in negative and that doesn't make sense

#We will drop those rows too



data = data[data.waiting_time_days >= 0]
#Check the data now

data['waiting_time_days'].unique()
data.info()
#We notice that the date type turned to object again, so I'll change it back



data['scheduledday'] = data['scheduledday'].astype('datetime64[ns]')



data['appointmentday'] = data['appointmentday'].astype('datetime64[ns]')
#Now we will get the weekday for scheduledday to see later if the weekday make a difference or not



data['scheduled_weekday'] = data['scheduledday'].apply(lambda time: time.dayofweek)
data.scheduled_weekday.unique()
data.head()
data.shape
data.info()
#We have here three columns (gender, neighbourhood, no_show) with object type that need to e changed to str

#So we can transform them with label encoder



data['gender'] = data['gender'].astype(str)



data['neighbourhood'] = data['neighbourhood'].astype(str)



data['no_show'] = data['no_show'].astype(str)
#Applying LabelEncoder on the following Features [gender, neighbourhood, no_show]



Encoder = LabelEncoder()



data['gender'] = Encoder.fit_transform(data['gender'])



data['neighbourhood'] = Encoder.fit_transform(data['neighbourhood'])



data['no_show'] = Encoder.fit_transform(data['no_show'])
#Now I'll drop [scheduledday, appointmentday] features, the give us no valuable info anymore



data = data.drop(['scheduledday', 'appointmentday'] , axis = 1, inplace = False)
#Checking the new data shape

data.shape
#Now we could proceed to classifications models

#First we will split features from output



#X Data

X = data.drop(['no_show'], axis = 1, inplace = False)



#y Data

y = data['no_show']
#Features shape



X.shape
#We'll try three normalizer to see what fit data the best



#Standard Scaler for Data



#scaler = StandardScaler(copy = True, with_mean = True, with_std = True)



#X = scaler.fit_transform(X)
#MinMaxScaler for Data



scaler = MinMaxScaler(copy = True, feature_range = (0, 1))



X = scaler.fit_transform(X)



#We foumd that min max scaler is the beat scaler to use with my data
#Normalizing Data



#scaler = Normalizer(copy = True, norm = 'l2') 



#X = scaler.fit_transform(X)
#Feature Selection by Percentile



FeatureSelection = SelectPercentile(score_func = f_classif, percentile = 50)



X = FeatureSelection.fit_transform(X, y)

#Splitting data (0.75 for train, 0.25 for test)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 44, shuffle = True)
#Creating an empty list to store models' accurcy for comparison purpose



Accuracy_results = []
#Applying LogisticRegression Model 





LogisticRegressionModel = LogisticRegression(penalty = 'l2', solver = 'sag', C = 1.0, random_state = 44)



LogisticRegressionModel.fit(X_train, y_train)





#Calculating Prediction

y_pred = LogisticRegressionModel.predict(X_test)



print('\n Logistic Regresiion Model Metrics: \n')





#Calculating Confusion Matrix

Confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', Confusion_matrix)

print('')



#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

Accuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy Score is : ', Accuracy_score)

print('')



#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))  

Recall_score = recall_score(y_test, y_pred, average = 'micro')

print('Recall Score is : ', Recall_score)

print('')



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

Precision_score = precision_score(y_test, y_pred, average = 'micro') 

print('Precision Score is : ', Precision_score)

print('')





Accuracy_results.append({'Logistic Regression Model': np.round(Accuracy_score, 3)})
#Applying SVC Model 



SVCModel = SVC(kernel= 'rbf', max_iter = 10000, C = 0.01, gamma = 'auto')



SVCModel.fit(X_train, y_train)



#Calculating Prediction

y_pred = SVCModel.predict(X_test)



print('\n SVC Model Metrics: \n')



#Calculating Confusion Matrix

Confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', Confusion_matrix)

print('')



#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

Accuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy Score is : ', Accuracy_score)

print('')



#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))  

Recall_score = recall_score(y_test, y_pred, average = 'micro')

print('Recall Score is : ', Recall_score)

print('')



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

Precision_score = precision_score(y_test, y_pred, average = 'micro') 

print('Precision Score is : ', Precision_score)





Accuracy_results.append({'SVC Model': np.round(Accuracy_score, 3)})

#Applying DecisionTreeClassifier Model 



DecisionTreeClassifierModel = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 44) 



DecisionTreeClassifierModel.fit(X_train, y_train)





#Calculating Prediction

y_pred = DecisionTreeClassifierModel.predict(X_test)



print('\n Decision Tree Classifier Model Metrics: \n')



#Calculating Confusion Matrix

Confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', Confusion_matrix)

print('')



#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

Accuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy Score is : ', Accuracy_score)

print('')



#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))  

Recall_score = recall_score(y_test, y_pred, average = 'micro')

print('Recall Score is : ', Recall_score)

print('')



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

Precision_score = precision_score(y_test, y_pred, average = 'micro') 

print('Precision Score is : ', Precision_score)





Accuracy_results.append({'Decision Tree Classifier Model': np.round(Accuracy_score, 3)})
#Applying RandomForestClassifier Model 



RandomForestClassifierModel = RandomForestClassifier(criterion = 'gini', n_estimators = 100,

                                                     max_depth = 3,random_state = 44) 



RandomForestClassifierModel.fit(X_train, y_train)





#Calculating Prediction

y_pred = RandomForestClassifierModel.predict(X_test)



print('\n Random Forest Classifier Model Metrics: \n')



#Calculating Confusion Matrix

Confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', Confusion_matrix)

print('')



#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

Accuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy Score is : ', Accuracy_score)

print('')



#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))  

Recall_score = recall_score(y_test, y_pred, average = 'micro')

print('Recall Score is : ', Recall_score)

print('')



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

Precision_score = precision_score(y_test, y_pred, average = 'micro') 

print('Precision Score is : ', Precision_score)



Accuracy_results.append({'Random Forest Classifier Model': np.round(Accuracy_score, 3)})
#Applying MLPClassifier Model 



MLPClassifierModel = MLPClassifier(activation = 'tanh', solver = 'lbfgs', learning_rate = 'constant',

                                   early_stopping = False,alpha = 0.0001,

                                   hidden_layer_sizes = (100, 3),random_state = 44)



MLPClassifierModel.fit(X_train, y_train)



#Calculating Prediction

y_pred = MLPClassifierModel.predict(X_test)



print('\n MLPClassifier Model Metrics: \n')



#Calculating Confusion Matrix

Confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', Confusion_matrix)

print('')



#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

Accuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy Score is : ', Accuracy_score)

print('')



#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))  

Recall_score = recall_score(y_test, y_pred, average = 'micro')

print('Recall Score is : ', Recall_score)

print('')



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

Precision_score = precision_score(y_test, y_pred, average = 'micro') 

print('Precision Score is : ', Precision_score)



Accuracy_results.append({'MLP Classifier Model': np.round(Accuracy_score, 3)})
#Applying GradientBoostingClassifier Model 



GBCModel = GradientBoostingClassifier(n_estimators = 100, max_depth = 3, random_state = 44)



GBCModel.fit(X_train, y_train)



#Calculating Prediction

y_pred = GBCModel.predict(X_test)



print('\n Gradient BoostingClassifier Model Metrics: \n')



#Calculating Confusion Matrix

Confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', Confusion_matrix)

print('')





#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

Accuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy Score is : ', Accuracy_score)

print('')



#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))  

Recall_score = recall_score(y_test, y_pred, average = 'micro')

print('Recall Score is : ', Recall_score)

print('')





#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

Precision_score = precision_score(y_test, y_pred, average = 'micro') 

print('Precision Score is : ', Precision_score)



Accuracy_results.append({'GBC Model': np.round(Accuracy_score, 3)})
Accuracy_results
#Applying Grid Searching :  

'''

model_selection.GridSearchCV(estimator, param_grid, scoring=None,fit_params=None, n_jobs=None, iid=’warn’,

                             refit=True, cv=’warn’, verbose=0,pre_dispatch=‘2*n_jobs’, error_score=

                             ’raisedeprecating’,return_train_score=’warn’)



'''



SelectedModel = SVCModel



SelectedParameters = {'C':[0.001, 0.01, 0.1],'kernel':['linear','rbf']}





GridSearchModel = GridSearchCV(SelectedModel,SelectedParameters, cv = 2, return_train_score = True)



GridSearchModel.fit(X_train, y_train)

sorted(GridSearchModel.cv_results_.keys())



GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 

                                                               'params' , 'rank_test_score' , 'mean_fit_time']]



# Showing Results

print('Best Score is: ', GridSearchModel.best_score_)

print('')



print('Best Parameters are: ', GridSearchModel.best_params_)

print('')



print('Best Estimator is: ', GridSearchModel.best_estimator_)
#Applying SVC Model with GridSearch best parameters



SVCModel = SVC(kernel= 'linear', max_iter = 10000, C = 0.001, gamma = 'auto')



SVCModel.fit(X_train, y_train)



#Calculating Prediction

y_pred = SVCModel.predict(X_test)

print('\n SVC Model Metrics: \n')



#Calculating Confusion Matrix

Confusion_matrix = confusion_matrix(y_test, y_pred)

print('Confusion Matrix is : \n', Confusion_matrix)

print('')



#Calculating Accuracy Score  : ((TP + TN) / float(TP + TN + FP + FN))

Accuracy_score = accuracy_score(y_test, y_pred)

print('Accuracy Score is : ', Accuracy_score)

print('')



#Calculating Recall Score : (Sensitivity) (TP / float(TP + FN))  

Recall_score = recall_score(y_test, y_pred, average = 'micro')

print('Recall Score is : ', Recall_score)

print('')



#Calculating Precision Score : (Specificity) #(TP / float(TP + FP))  

Precision_score = precision_score(y_test, y_pred, average = 'micro') 

print('Precision Score is : ', Precision_score)

print('')



Accuracy_results.append({'SVCModel': np.round(Accuracy_score, 3)})

Accuracy_results
#Calculating Prediction using Gradient Boosting classifer model 



y_pred = GBCModel.predict(X_test)



print(y_test[0:10])



print('')



print(y_pred[:10])
#Use Encoder to go back to yes and now values



print(Encoder.inverse_transform(y_pred[:20]))



print('')



print(Encoder.inverse_transform(y_test[:20]))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
telecom_data = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
telecom_data.info()
telecom_data.head()
telecom_data = telecom_data.replace( { 'SeniorCitizen': { 0: 'No', 1:'Yes' } } )
def categorical_segment(column_name:str) -> 'grouped_dataframe':
    segmented_df = telecom_data[[column_name, 'Churn']]
    segmented_churn_df = segmented_df[segmented_df['Churn'] == 'Yes']
    grouped_df = segmented_churn_df.groupby(column_name).count().reset_index().rename(columns = {'Churn':'Churned'})
    total_count_df = segmented_df.groupby(column_name).count().reset_index().rename(columns = {'Churn':'Total'})
    merged_df = pd.merge(grouped_df, total_count_df, how = 'inner', on = column_name)
    merged_df['Percent_Churned'] = merged_df[['Churned','Total']].apply(lambda x: (x[0] / x[1]) * 100, axis=1) 
    return merged_df

categorical_columns_list = list(telecom_data.columns)[1:5] + list(telecom_data.columns)[6:18]

grouped_df_list = []

for column in categorical_columns_list:
    grouped_df_list.append( categorical_segment( column ) )
    
grouped_df_list[0]
import matplotlib.pyplot as plt 
for i , column in enumerate(categorical_columns_list):
    fig, ax = plt.subplots(figsize=(13,5))
    plt.bar(grouped_df_list[i][column] , [ 100 - i for i in grouped_df_list[i]['Percent_Churned'] ],width = 0.1, color = 'g')
    plt.bar(grouped_df_list[i][column],grouped_df_list[i]['Percent_Churned'], bottom =  [ 100 - i for i in grouped_df_list[i]['Percent_Churned'] ],
            width = 0.1, color = 'r')
    plt.title('Percent Churn by ' + column)
    plt.xlabel(column)
    plt.ylabel('Percent Churned')
    plt.legend( ('Retained', 'Churned') )
    plt.show()

def continous_var_segment(column_name:str) -> 'segmented_df':
    segmented_df = telecom_data[[column_name, 'Churn']]
    segmented_df = segmented_df.replace( {'Churn': {'No':'Retained','Yes':'Churned'} } )
    segmented_df['Customer'] = ''
    return segmented_df

continous_columns_list = [list(telecom_data.columns)[18]] + [list(telecom_data.columns)[5]]


continous_segment_list = []

for var in continous_columns_list:
    continous_segment_list.append( continous_var_segment(var) )
    
import seaborn as sns
sns.set('talk')

for i, column in enumerate( continous_columns_list ):
    fig, ax = plt.subplots(figsize=(8,11))
    sns.violinplot(x = 'Customer', y = column, data = continous_segment_list[i], hue = 'Churn', split = True)
    plt.title('Churn by ' + column)
    plt.show()

from sklearn.cluster import KMeans 
from sklearn.preprocessing import MinMaxScaler

monthlyp_and_tenure = telecom_data[['MonthlyCharges','tenure']][telecom_data.Churn == 'Yes']

scaler = MinMaxScaler()
monthly_and_tenure_standardized = pd.DataFrame( scaler.fit_transform(monthlyp_and_tenure) )
monthly_and_tenure_standardized.columns = ['MonthlyCharges','tenure']

kmeans = KMeans(n_clusters = 3, random_state = 42).fit(monthly_and_tenure_standardized)

monthly_and_tenure_standardized['cluster'] = kmeans.labels_

fig, ax = plt.subplots(figsize=(13,8))
plt.scatter( monthly_and_tenure_standardized['MonthlyCharges'], monthly_and_tenure_standardized['tenure'],
           c = monthly_and_tenure_standardized['cluster'], cmap = 'Spectral')

plt.title('Clustering churned users by monthly Charges and tenure')
plt.xlabel('Monthly Charges')
plt.ylabel('Tenure')


plt.show()


telecom_data_filtered = telecom_data.drop(['TotalCharges','customerID'], axis = 1)

def encode_binary(column_name:str):
    global telecom_data_filtered
    telecom_data_filtered = telecom_data_filtered.replace( { column_name: { 'Yes': 1 , 'No': 0 } }  )
    

binary_feature_list = list(telecom_data_filtered.columns)[1:4] + [list(telecom_data_filtered.columns)[5]] \
+ [list(telecom_data_filtered.columns)[15]]  \
+ [list(telecom_data_filtered.columns)[18]]
    
for binary_feature in binary_feature_list:
    encode_binary(binary_feature)
    

telecom_data_processed = pd.get_dummies( telecom_data_filtered, drop_first = True )

telecom_data_processed.head(10)


telecom_data.Churn.value_counts()
import numpy as np 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.metrics as metrics
%matplotlib inline 

X = np.array( telecom_data_processed.drop( ['Churn'] , axis = 1 ) )
y = np.array( telecom_data_processed['Churn'] )

X_train,X_test,y_train,y_test = train_test_split( X, y , test_size = 0.2 , random_state = 42 )

def get_metrics( model ):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    y_actual = y_test 
    print()
    print('-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
    print()
    print('Accuracy on unseen hold out set:' , metrics.accuracy_score(y_actual,y_pred) * 100 , '%' )
    print()
    f1_score = metrics.f1_score(y_actual,y_pred)
    precision = metrics.precision_score(y_actual,y_pred)
    recall = metrics.recall_score(y_actual,y_pred)
    score_dict = { 'f1_score':[f1_score], 'precision':[precision], 'recall':[recall]}
    score_frame = pd.DataFrame(score_dict)
    print(score_frame)
    print()
    fpr, tpr, thresholds = metrics.roc_curve( y_actual, y_prob[:,1] ) 
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot( fpr, tpr, 'b-', alpha = 0.5, label = '(AUC = %.2f)' % metrics.auc(fpr,tpr) )
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend( loc = 'lower right')
    plt.show()
rf = RandomForestClassifier( n_estimators = 20, n_jobs=-1, max_features = 'sqrt', random_state = 42 )
param_grid1 = {"min_samples_split": np.arange(2,11), 
              "min_samples_leaf": np.arange(1,11)}
rf_cv = GridSearchCV( rf, param_grid1, cv=5, iid = False )
rf_cv.fit(X_train,y_train)
print( rf_cv.best_params_ )
print( rf_cv.best_score_ )
rf = RandomForestClassifier( n_estimators = 20, n_jobs = -1, min_samples_split = 2, min_samples_leaf = 8, random_state = 2 )
param_grid2 = {'max_depth': np.arange(9,21),
              'max_features':['sqrt','log2']}
rf_cv = GridSearchCV(rf, param_grid2, cv=5, iid = False)
rf_cv.fit(X_train,y_train)
print( rf_cv.best_params_ )
print( rf_cv.best_score_ )
rf = RandomForestClassifier( n_estimators = 1000, max_features = 'log2', max_depth = 11, min_samples_split = 2, 
                          min_samples_leaf = 8, n_jobs = -1 , random_state = 42, class_weight = {0:0.95, 1:2})
rf.fit(X_train,y_train)
print('Training Accuracy:',rf.score(X_train,y_train)*100,'%')
get_metrics(rf)
model_pipeline = Pipeline( steps = [( 'normalizer', MinMaxScaler() ), 
                                   ( 'log_reg', LogisticRegression( penalty = 'l2', random_state = 42 ) ) ] )
param_dict = dict( log_reg__C = [0.001, 0.01, 0.1, 1, 10, 100])
estimator = GridSearchCV( model_pipeline, param_grid = param_dict, cv = 5, n_jobs = -1, iid = False )
estimator.fit(X_train,y_train)
print(estimator.best_params_)
print(estimator.best_score_)
model_pipeline = Pipeline( steps = [( 'normalizer', MinMaxScaler() ), 
                                   ( 'log_reg', LogisticRegression( penalty = 'l2', C = 100, random_state = 42, class_weight = {0:.95 , 1:2} ) ) ] )
model_pipeline.fit(X_train,y_train)
print('Training Accuracy:',model_pipeline.score(X_train,y_train)*100,'%')
get_metrics(model_pipeline)
svc_pipeline = Pipeline( steps = [( 'normalizer', MinMaxScaler() ), 
                                   ( 'svc', SVC(random_state = 42, probability = True) ) ] )
params = [0.001, 0.01, 0.1, 1, 10]
param_dict = dict( svc__C = params, svc__gamma = params)
estimator = GridSearchCV( svc_pipeline, param_grid = param_dict, cv = 5, n_jobs = -1, iid = False )
estimator.fit(X_train,y_train)
print(estimator.best_params_)
print(estimator.best_score_)
svc = SVC(C = 1, gamma = 0.01, class_weight = {0:1, 1:2}, random_state = 42, probability = True)
svc.fit(X_train,y_train) 
print('Training Accuracy:',svc.score(X_train,y_train)*100,'%')
get_metrics(svc)
coeff = model_pipeline.named_steps['log_reg'].coef_.flatten()
coeff_updated = np.append( coeff[:8], [sum(coeff[8:10]), sum(coeff[10:12]), sum(coeff[12:14]),sum(coeff[14:16]), 
                         sum(coeff[16:18]), sum(coeff[18:20]), sum(coeff[20:22]), sum(coeff[22:24]), sum(coeff[24:26]), sum(coeff[26:]) ] )
columns = ['SeniorCitizen', 'Partner', 'Dependents', 'Tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'Gender', 'MultipleLines',
          'InternetService', 'OnlineSecurity','OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV','StreamingMovies', 'Contract', 'PaymentMethod']
fig, ax = plt.subplots(figsize=(50,20))
plt.plot(columns, coeff_updated, c = 'yellow', marker='o', linewidth = 6, linestyle='dashed', markersize=20, mfc = 'red')
plt.title('Coefficients Learned by the Logistic Regression Model')
plt.ylabel('Coeff')
plt.xlabel('Features')
plt.show()

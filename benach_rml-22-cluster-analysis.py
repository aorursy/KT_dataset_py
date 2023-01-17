# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.




# Data Preprocessing - Data import

os.chdir(r'/kaggle/input/telco-customer-churn')

My_data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

My_data.head()



print(My_data.describe())

My_data.dtypes
#Data Cleansing and Analysis

M_Values = My_data.isnull()

M_Values.head()
#Visualize the missing data

import seaborn as sns

sns.heatmap(data = M_Values, yticklabels=False, cbar=False, cmap='viridis')
# replace values for SeniorCitizen as a categorical feature

My_data['SeniorCitizen'] = My_data['SeniorCitizen'].replace({1:'Yes',0:'No'})

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

My_data[num_cols].describe()
def categorical_segment(column_name:str) -> 'grouped_dataframe':

    segmented_df = My_data[[column_name, 'Churn']]

    segmented_churn_df = segmented_df[segmented_df['Churn'] == 'Yes']

    grouped_df = segmented_churn_df.groupby(column_name).count().reset_index().rename(columns = {'Churn':'Churned'})

    total_count_df = segmented_df.groupby(column_name).count().reset_index().rename(columns = {'Churn':'Total'})

    merged_df = pd.merge(grouped_df, total_count_df, how = 'inner', on = column_name)

    merged_df['Percent_Churned'] = merged_df[['Churned','Total']].apply(lambda x: (x[0] / x[1]) * 100, axis=1) 

    return merged_df



categorical_columns_list = list(My_data.columns)[1:5] + list(My_data.columns)[6:18]



grouped_df_list = []



for column in categorical_columns_list:

    grouped_df_list.append( categorical_segment( column ) )

    

grouped_df_list[0]
#Churn by categorical features

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
#Churn by numerical features

def continous_var_segment(column_name:str) -> 'segmented_df':

    segmented_df = My_data[[column_name, 'Churn']]

    segmented_df = segmented_df.replace( {'Churn': {'No':'Retained','Yes':'Churned'} } )

    segmented_df['Customer'] = ''

    return segmented_df



continous_columns_list = [list(My_data.columns)[18]] + [list(My_data.columns)[5]]





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



# Normalizing  tenure and monthly charges and using K-means clustering to cluster churned customers based on them.



from sklearn.cluster import KMeans 

from sklearn.preprocessing import MinMaxScaler



monthlyp_and_tenure = My_data[['MonthlyCharges','tenure']][My_data.Churn == 'Yes']



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



#Pre-processing the data using label encoding and one hot encoding to get it ready for ML Model



My_data_filtered = My_data.drop(['TotalCharges','customerID'], axis = 1)



def encode_binary(column_name:str):

    global My_data_filtered

    My_data_filtered = My_data_filtered.replace( { column_name: { 'Yes': 1 , 'No': 0 } }  )

    



binary_feature_list = list(My_data_filtered.columns)[1:4] + [list(My_data_filtered.columns)[5]] \

+ [list(My_data_filtered.columns)[15]]  \

+ [list(My_data_filtered.columns)[18]]

    

for binary_feature in binary_feature_list:

    encode_binary(binary_feature)

    



My_data_processed = pd.get_dummies( My_data_filtered, drop_first = True )



My_data_processed.head(10)



My_data.Churn.value_counts()
# Importing all the necessary librabries

import numpy as np 

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import sklearn.metrics as metrics

%matplotlib inline 



X = np.array( My_data_processed.drop( ['Churn'] , axis = 1 ) )

y = np.array( My_data_processed['Churn'] )



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
#Model 1: Random Forest 

rf = RandomForestClassifier( n_estimators = 20, n_jobs=-1, max_features = 'sqrt', random_state = 42 )

param_grid1 = {"min_samples_split": np.arange(2,11), 

              "min_samples_leaf": np.arange(1,11)}

rf_cv = GridSearchCV( rf, param_grid1, cv=5, iid = False )

rf_cv.fit(X_train,y_train)

print( rf_cv.best_params_ )

print( rf_cv.best_score_ )
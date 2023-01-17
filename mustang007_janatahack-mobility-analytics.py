# Importing Important libraries

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
# reading datafiles

train = pd.read_csv('/kaggle/input/mobility-analytics-dataset/train_Wc8LBpr.csv')
test = pd.read_csv('/kaggle/input/mobility-analytics-dataset/test_VsU9xXK.csv')
sub = pd.read_csv('/kaggle/input/mobility-analytics-dataset/sample_submission_NoPBkjr.csv')
train.head()
train['Type_of_Cab'].isnull().sum()
# method to check null values 
def missing_values_check(data):
    for i in data.columns:
        null_value = data[i].isnull().sum()
        if (null_value) > 0:
            print(i,' has ==', null_value)
# Check Null Values in train dataset
missing_values_check(train)
# Check Null Values in test dataset
missing_values_check(test)
# data information
train.info()
for i in train.describe():
    print(i)
# lets see categorical and non categorical columns
cat = []
no_cat = []
def cat_no_cat(data):
    for i in data.describe():
        no_cat.append(i)
    for i in data.describe(include='O'):
        cat.append(i)
    return no_cat, cat
    
no_cat_col, cat_col = cat_no_cat(train)
print('list of category columns \n ',cat_col)
print('list of non_category columns \n ',no_cat_col,'\n')

no_cat_data = train[no_cat_col]
cat_data = train[cat_col]
# Check Null Values in non_category dataset
missing_values_check(no_cat_data) 
# Lets impute null values with K-means method
def null_imputer(data):
    knnimputer = KNNImputer()
    for i in data.columns:
        a = np.array(data[i]).reshape(-1,1)
        data[i] = knnimputer.fit_transform(a)
    return data    
# this method is only for non category colums
no_cat_data = null_imputer(no_cat_data)
# Check Null Values in category dataset
missing_values_check(cat_data)
cat_data = cat_data.fillna(-999)
cat_data
cat_data['Gender'].value_counts()
cat_data.drop(columns='Trip_ID', inplace=True)
# to calculate unique values
def unique_values(data):
    for i in data:
        print(i, ' has =',data[i].value_counts())
unique_values(cat_data)
sample_data = pd.DataFrame()
for i in cat_data:
    a = pd.get_dummies(cat_data[i],prefix=i)
    a = pd.DataFrame(a)
    sample_data = pd.concat([a,sample_data], axis=1)
sample_data
no_cat_data.head()
combine_data = pd.concat([no_cat_data, sample_data], axis=1)
combine_data.head()
# Lets split our data
X = combine_data.drop(columns=['Surge_Pricing_Type'])
Y = combine_data['Surge_Pricing_Type']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2)
base_model = RandomForestClassifier()
base_model.fit(x_train, y_train)
from sklearn.metrics import accuracy_score
prediction = base_model.predict(x_test)
accuracy_score(prediction, y_test)
# let see most important Features
imp = base_model.feature_importances_
top_imp = [(a*100)/sum(imp) for a in imp]
feature_importance = pd.DataFrame()
feature_importance['features'] = x_train.columns
feature_importance['% imp'] = top_imp
feature_importance.sort_values(by ='% imp',ascending=False).reset_index(drop=True)

# lets combine var1, var 2, var3 n take their mean
combine_data['average_var'] =  (combine_data['Var1'] + combine_data['Var2'] + combine_data['Var3'])/3

feature_extraction = pd.concat([no_cat_data, cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Destination_Type'])['Trip_Distance'].mean().to_frame().reset_index()
dest_trip = cat_data.merge(dest_trip, on='Destination_Type') 
dest_trip.head()
# lets add Trip_distance_mean in combine data
combine_data['Trip_Distance_mean'] = dest_trip['Trip_Distance']
combine_data.head()
feature_extraction = pd.concat([no_cat_data, cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Type_of_Cab'])['Trip_Distance'].mean().to_frame().reset_index()
dest_trip = cat_data.merge(dest_trip, on='Type_of_Cab') 
dest_trip.head()
# lets add Trip_cab_mean in combine data
combine_data['Trip_cab_mean'] = dest_trip['Trip_Distance']
combine_data.head()
feature_extraction = pd.concat([no_cat_data, cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Customer_Since_Months'])['Trip_Distance'].mean().to_frame().reset_index()
dest_trip = no_cat_data.merge(dest_trip, on='Customer_Since_Months') 
dest_trip.head()
# lets add Trip_cab_mean in combine data
combine_data['Trip_customer_months_mean'] = dest_trip['Trip_Distance_y']
combine_data.head()
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
bin_array = np.array(combine_data['Trip_Distance']).reshape(-1,1)
combine_data['Trip_distance_bins'] = est.fit_transform(bin_array)
feature_extraction = pd.concat([no_cat_data, cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Type_of_Cab'])['Customer_Rating'].mean().to_frame().reset_index()
dest_trip = cat_data.merge(dest_trip, on='Type_of_Cab') 
dest_trip.head()
# lets add Customer_rating_distance_mean in combine data
combine_data['Customer_Distance_mean'] = dest_trip['Customer_Rating']
combine_data.head()
# Lets split our data
X = combine_data.drop(columns=['Surge_Pricing_Type'])
Y = combine_data['Surge_Pricing_Type']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2)

base_model = RandomForestClassifier()
base_model.fit(x_train, y_train)


prediction = base_model.predict(x_test)
accuracy_score(prediction, y_test)

# let see most important Features
imp = base_model.feature_importances_
top_imp = [(a*100)/sum(imp) for a in imp]
feature_importance = pd.DataFrame()
feature_importance['features'] = x_train.columns
feature_importance['% imp'] = top_imp
feature_importance.sort_values(by ='% imp',ascending=False).reset_index(drop=True)
for i in combine_data.columns:
    cor_colum = combine_data[i].corr(combine_data['Surge_Pricing_Type'])
    if cor_colum > 0.8:
        print(i,' has = ',cor_colum)
    elif cor_colum < -0.8:
        print(i,' has = ',cor_colum)
        
        
# No Columns is Highly correlated to each other
lgbm_model = LGBMClassifier()
lgbm_model.fit(x_train, y_train)
prediction = lgbm_model.predict(x_test)
accuracy_score(prediction, y_test)
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(x_train, y_train)
prediction = xgb_model.predict(x_test)
accuracy_score(prediction, y_test)
! pip install pycaret
from pycaret.classification import *
session_1 = setup(data=combine_data, target='Surge_Pricing_Type', log_experiment=True)
best_model = create_model('lightgbm')
tuned_gbm_model = tune_model(best_model)
# with tuned model
tuned_gbm_model.fit(x_train, y_train)
prediction = tuned_gbm_model.predict(x_test)
accuracy_score(prediction, y_test)
# create xgboost
xgb = create_model('xgboost')
# Lets combine both models
blend = blend_models(estimator_list = [xgb, tuned_gbm_model], method='soft')
# lets see categorical and non categorical columns of test data
test_cat = []
test_no_cat = []
def cat_no_cat(data):
    for i in data.describe():
        test_no_cat.append(i)
    for i in data.describe(include='O'):
        test_cat.append(i)
    return test_no_cat, test_cat
test_no_cat_col, test_cat_col = cat_no_cat(test)
print('list of category columns \n ',test_cat_col)
print('list of non_category columns \n ',test_no_cat_col,'\n')
test_no_cat_data = test[test_no_cat_col]
test_cat_data = test[test_cat_col]
# Check Null Values in non_category test dataset
missing_values_check(test_no_cat_data) 
# this method is only for non category colums
test_no_cat_data = null_imputer(test_no_cat_data)
test_cat_data = test_cat_data.fillna(-999)
test_cat_data.drop(columns='Trip_ID', inplace=True)
sample_data = pd.DataFrame()
for i in test_cat_data:
    a = pd.get_dummies(test_cat_data[i],prefix=i)
    a = pd.DataFrame(a)
    sample_data = pd.concat([a,sample_data], axis=1)

combine_data = pd.DataFrame()
combine_data = pd.concat([test_no_cat_data, sample_data], axis=1)
combine_data.head()
# lets combine var1, var 2, var3 n take their mean
combine_data['average_var'] =  (combine_data['Var1'] + combine_data['Var2'] + combine_data['Var3'])/3
feature_extraction = pd.concat([test_no_cat_data, test_cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Destination_Type'])['Trip_Distance'].mean().to_frame().reset_index()
dest_trip = test_cat_data.merge(dest_trip, on='Destination_Type') 
dest_trip.head()
# lets add Trip_distance_mean in combine test data
combine_data['Trip_Distance_mean'] = dest_trip['Trip_Distance']
combine_data.head()
feature_extraction = pd.concat([test_no_cat_data, test_cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Type_of_Cab'])['Trip_Distance'].mean().to_frame().reset_index()
dest_trip = test_cat_data.merge(dest_trip, on='Type_of_Cab') 
dest_trip.head()
combine_data['Trip_cab_mean'] = dest_trip['Trip_Distance']
combine_data.head()
feature_extraction = pd.concat([test_no_cat_data, test_cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Customer_Since_Months'])['Trip_Distance'].mean().to_frame().reset_index()
dest_trip = test_no_cat_data.merge(dest_trip, on='Customer_Since_Months') 
combine_data['Trip_customer_months_mean'] = dest_trip['Trip_Distance_y']
combine_data.head()
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
bin_array = np.array(combine_data['Trip_Distance']).reshape(-1,1)
combine_data['Trip_distance_bins'] = est.fit_transform(bin_array)
feature_extraction = pd.concat([test_no_cat_data, test_cat_data], axis=1)
dest_trip = feature_extraction.groupby(['Type_of_Cab'])['Customer_Rating'].mean().to_frame().reset_index()
dest_trip = test_cat_data.merge(dest_trip, on='Type_of_Cab') 
combine_data['Customer_Distance_mean'] = dest_trip['Customer_Rating']
combine_data.head()
# LEts check submission format
sub
# generate predictions on unseen data
predictions = predict_model(blend, data = combine_data)
sub['Surge_Pricing_Type'] = predictions['Label']
sub.to_csv('Final_blend.csv')
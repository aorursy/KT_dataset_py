# Reading the input directory files
import os
print(os.listdir("../input/"))
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
%matplotlib inline
# Reading the Bank Customers file using pandas function read.csv()
customers_data = pd.read_csv('../input/Churn_Modelling.csv')
# Displaying the top rows of the dataset for a quick visualization of the data
print(customers_data.head())
# running a script on customers data  file: customers_data.describe() to run the descriptive statistics on the data
#in order to screen outliers and potential bad data.

customers_data.describe(include="all")
# analyzing the data, to know the number of rows and columns and see if there are any missing data
customers_data.shape
print(" The number of null values is: " , customers_data.isnull().values.sum())
print(customers_data.isnull().sum())
# Running customers_data.info () command to check if there are no missing values in any of the fields or NaN 
# and if all columns types were consistent with the data they contains. All were complete and consistent.
customers_data.info () 

#Creating helper functions to see visualy the distributon of the the different predictor variables

def visual_exploratory(x):
    
    for var in x.select_dtypes(include = [np.number]).columns :
        print( var + ' : ')
        x[var].plot('hist')
        plt.show()
        
visual_exploratory(customers_data)

# ploting the box plot to visually inspect numeric data

def boxPlot_exploratory(x):
    
    for var in x.select_dtypes(include = [np.number]).columns :
        print( var + ' : ')
        x.boxplot(column = var)
        plt.show()
        
boxPlot_exploratory(customers_data)
#Creating a variable of Categorical features

cat_df_customers = customers_data.select_dtypes(include = ['object']).copy()
print(cat_df_customers.head()) 
print(" The number of null values is: " , cat_df_customers.isnull().values.sum())
#Plotting categorical features

## 1. Plot for Geographical location

location_count = cat_df_customers['Geography'].value_counts()
sns.barplot(location_count.index, location_count.values)
plt.title('Geographical location Distribution of Bank Customers')
plt.ylabel('Frequency', fontsize=11)
plt.xlabel('Geography', fontsize=11)
plt.show()


## 2. Plot for Gender 

location_count = cat_df_customers['Gender'].value_counts()
sns.barplot(location_count.index, location_count.values)
plt.title('Gender Distribution of Bank Customers')
plt.ylabel('Frequency', fontsize=11)
plt.xlabel('Gender', fontsize=11)
plt.show()
#gradient boosting decision tree algorithm
import xgboost as xgb
import sklearn as skt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
new_customers_data=customers_data.copy()
# encode string class values as integers

Gender = new_customers_data['Gender']
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Gender)
label_encoded =label_encoder.transform(Gender)
new_customers_data['Gender']=label_encoded
#print(new_customers_data.head())
#Gend = new_customers_data['Gender']
#print(Gend)
temp_customers_data=new_customers_data.copy()
temp_customers_data = pd.get_dummies(temp_customers_data, columns=['Geography'], prefix = ['Geography'])
print(temp_customers_data.head())
# Appending the new column to the new_customers_data dataframe

new_customers_data.insert(13, 'Geography_France' , temp_customers_data['Geography_France'])
new_customers_data.insert(14, 'Geography_Germany' , temp_customers_data['Geography_Germany'])
new_customers_data.insert(15, 'Geography_Spain' , temp_customers_data['Geography_Spain'])
print(new_customers_data.head())
# Helper function that will create and add a new column tof credit score range the data frame
def creditscore(data):
    score = data.CreditScore
    score_range =[]
    for i in range(len(score)) : 
        if (score[i] < 600) :  
            score_range.append(1) # 'Very Bad Credit'
        elif ( 600 <= score[i] < 650) :  
            score_range.append(2) # 'Bad Credit'
        elif ( 650 <= score[i] < 700) :  
            score_range.append(3) # 'Good Credit'
        elif ( 700 <= score[i] < 750) :  
            score_range.append(4) # 'Very Good Credit'
        elif score[i] >= 750 : 
            score_range.append(5) # 'Excellent Credit'
    return score_range

# converting the returned list into a dataframe
CreditScore_category = pd.DataFrame({'CreditScore_range': creditscore(new_customers_data)})

# Appending the new column to the new_customers_data dataframe
new_customers_data.insert(16, 'CreditScore_range' , CreditScore_category['CreditScore_range'])
# Helper function that will create and add a new column of age group to the data frame
def agegroup(data):
    age = data.Age
    age_range =[]
    for i in range(len(age)) : 
        if (age[i] < 30) :  
            age_range.append(1) # 'Between 18 and 30 year'   
        elif ( 30 <= age[i] < 40) :  
            age_range.append(2) # 'Between 30 and 40 year'
        elif ( 40 <= age[i] < 50) :  
            age_range.append(3) # 'Between 40 and 50 year'
        elif ( 50 <= age[i] < 60) :  
            age_range.append(4) # ''Between 50 and 60 year'
        elif ( 60 <= age[i] < 70) :  
            age_range.append(5) # 'Between 60 and 70 year'
        elif ( 70 <= age[i] < 80) :  
            age_range.append(6) # 'Between 70 and 80 year'
        elif age[i] >= 80 : 
            age_range.append(7) # ''Above 80 year'
    return age_range

# converting the returned list into a dataframe
AgeGroup_category = pd.DataFrame({'age_group': agegroup(new_customers_data)})

# Appending the new column to the new_customers_data dataframe
new_customers_data.insert(17, 'age_group' , AgeGroup_category['age_group'])
print(new_customers_data.head())
new_customers_data_xgboost=new_customers_data.copy()
Target = 'Exited'
Surname = 'Surname'
Geography = 'Geography'
#Gender= 'Gender'
ID= 'RowNumber'
CustomerId = 'CustomerId'
#Choose all predictors except Target, Surname, Geography, CustomerId & ID and also separate the response variable
X = [x for x in new_customers_data_xgboost.columns if x not in [Surname,Geography, Target, ID, CustomerId]]
Y = new_customers_data_xgboost.iloc[:,-1]

predictors = new_customers_data_xgboost[X] #predictor variable
response = Y # response variable
print(predictors.head())
print(response.head())
# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(predictors, response, test_size=test_size,
random_state=seed)
# fit model on training data

## xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
##                max_depth = 5, alpha = 10, n_estimators = 10)

#model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                #max_depth = 5, alpha = 10, n_estimators = 10)
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
# make predictions for test data
predictions = model.predict(X_test)
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
# plotting important features for a quick idea of which contribute to the model perfromance better
import matplotlib.pyplot as plt
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold # to nforce the same distribution of classes in each fold
from sklearn.model_selection import cross_val_score
# testing the cross validated model
model2 = xgb.XGBClassifier()
kfold = StratifiedKFold(n_splits=10, random_state=7)
results = cross_val_score(model2, predictors, response, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print(results) # These results represent the accuracy at each fold in the cross validated model
print(model.feature_importances_) # the inbuild method from the model  display the importance score according to the input order of the predictors
# plot feature importance
from matplotlib import pyplot
xgb.plot_importance(model)
pyplot.show()
print(np.sort(model.feature_importances_)) # Sorting them according to the importance order of the features
from sklearn.feature_selection import SelectFromModel
thresholds = np.sort(model.feature_importances_)
for thresh in thresholds:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = xgb.XGBClassifier()
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],accuracy*100.0))
from sklearn.feature_selection import SelectFromModel
# select features using threshold
thresh= 0.04582651

selection = SelectFromModel(model, threshold=thresh, prefit=True)
select_X_train = selection.transform(X_train)
# train model
selection_model = xgb.XGBClassifier()
selection_model.fit(select_X_train, y_train)
# eval model
select_X_test = selection.transform(X_test)
y_pred = selection_model.predict(select_X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1],accuracy*100.0))
pred = pd.DataFrame(y_pred)
print (pred.head())
with open('churns_predict.csv', 'w') as f:
    print( pred, file=f) 
predictions = [round(value) for value in y_pred]
print(predictions)
preds = pd.DataFrame(predictions)
print(preds)


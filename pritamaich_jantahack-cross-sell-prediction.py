import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
data = pd.read_csv('../input/jantahack-cross-sell-prediction/train.csv')
data.head()
test_data = pd.read_csv('../input/jantahack-cross-sell-prediction/test.csv')
test_data.head()
data.info()
data.describe()
test_data.info()
test_data.describe()
#Creating copies of datassets

train = data.copy()
test = test_data.copy()
plt.figure(figsize=(12,5))
sns.countplot(train['Response'])
plt.title('Customer responses', fontsize = 15)
plt.show()
#Calculating positive and negative samples in percentages

positive_percent = len(train[train['Response'] == 1])/len(data)*100 
negative_percent = len(train[train['Response'] == 0])/len(data)*100
percentages = [positive_percent,negative_percent]

# Creating a pie chart
plt.figure(figsize = (6,8))
plt.pie(percentages, labels=['Interested','Not Interested'], autopct = '%.1f%%', colors = ['#aecc35', 'cyan'])
plt.title('Positive and negative samples (Percentages)', fontsize = 18)
plt.show()
plt.figure(figsize = (12,5))
sns.distplot(train['Age'], kde = False, color = 'Red')
plt.xlabel('Age', fontsize = 15)
plt.title('Age distribution', fontsize = 20)
plt.show()
plt.figure(figsize = (10,6) )
sns.countplot(data = train, x = 'Gender', hue = 'Response', palette = 'nipy_spectral')
plt.xlabel('Gender', fontsize = 14)
plt.ylabel('Count', fontsize = 14)
plt.title('Gender count according to Response', fontsize = 18)
plt.show()
train['Driving_License'].value_counts()
plt.figure(figsize = (10,5) )
sns.countplot(train['Driving_License'])
plt.xlabel('Driving License', fontsize = 12)
plt.show()
plt.figure(figsize = (10,6) )
sns.countplot(data = train, x = 'Vehicle_Age', hue = 'Response', palette='CMRmap_r')
plt.xlabel('Vehicle Age', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.title('Vehicle Age and Customer Response analysis', fontsize = 19)
plt.show()
vehicles = train[train['Vehicle_Age'] == '1-2 Year']

plt.figure(figsize = (10,6) )
sns.countplot(data = vehicles, x = 'Vehicle_Damage', hue = 'Response', palette='Set1')
plt.xlabel('Vehicle Damage status', fontsize = 15)
plt.ylabel('Number of vehicle damaged', fontsize = 15)
plt.title('Customer response analysis according vehicle damage status', fontsize = 19)
plt.show()
plt.figure(figsize=(15,6))
sns.kdeplot(data['Annual_Premium'])
plt.xlabel('Annual Premium', fontsize = 14)
plt.title('Annua Premium distribution', fontsize = 18)
plt.show()
plt.figure(figsize = (12,5))
sns.distplot(train['Policy_Sales_Channel'], kde = False, color = 'orange')
plt.xlabel('Policy Sales Chanel', fontsize = 15)
plt.title('Policy Sales distribution', fontsize = 20)
plt.show()
plt.figure(figsize = (12,5))
sns.distplot(train['Region_Code'], kde = False, color = 'blue')
plt.xlabel('Region Code', fontsize = 15)
plt.title('Region Code distribution', fontsize = 20)
plt.show()
plt.figure(figsize = (10,6) )
sns.countplot(data = train, x = 'Previously_Insured', hue = 'Response', palette='rainbow')
plt.xlabel('Previously Insured', fontsize = 15)
plt.ylabel('Count', fontsize = 15)
plt.title('Customer response based on previous insurance', fontsize = 19)
plt.show()
plt.figure(figsize = (18,8))
sns.heatmap(train.corr(), cmap = 'twilight', annot = True)
plt.show()
plt.figure(figsize=(12,5))
train.corrwith(train['Response']).sort_values().drop('Response').plot(kind='bar', color = 'red')
plt.title('Correlation with Customer Response (bar plot) ', fontsize= 15)
plt.show()
train_data = pd.read_csv('../input/jantahack-cross-sell-prediction/train.csv')
test_data = pd.read_csv('../input/jantahack-cross-sell-prediction/test.csv')
train = train_data.copy()
test = test_data.copy()
def preprocessing(data):
    
    #Dropping null values
    data = data.dropna()
    
    #Dropping id
    data = data.drop('id', axis = 1)
    
    #Columns to get dummies
    cols = ['Gender', 'Vehicle_Damage', 'Vehicle_Age']
    
    #Changing categories into dummies
    data_dum = pd.get_dummies(data = data , columns = cols, drop_first = True )
    
    #We don't need this column as it has almost no correlation with our dependent variable
    data_dum = data_dum.drop('Vintage', axis = 1)
    
    return data_dum
#Preprocessing training data
train_dum = preprocessing(train)

#Preprocessing test data
test_dum = preprocessing(test)
#Assigning inputs and targets

inputs = train_dum.drop('Response', axis = 1)
targets = train_dum['Response'] 

x_test = test_dum.copy()
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits = 10, random_state = 42)

for train_idx, val_idx in skf.split(inputs, targets):
    x_train, x_val = inputs.iloc[train_idx], inputs.iloc[val_idx]
    y_train, y_val = targets.iloc[train_idx], targets.iloc[val_idx]
#Scaling all input data
    
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)

#Will use only transform for validation and test data as we don't want any data leakage
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

lgbm = LGBMClassifier(num_leaves = 30, max_depth = 5, n_estimators = 550, learning_rate = 0.05, objective = 'binary', 
                      lambda_l2 = 12,
                      max_bin = 100, metric = 'auc', is_unbalance = True, random_state = None, n_jobs = -1)
lgbm.fit(x_train_scaled,y_train)
y_val_pred = lgbm.predict_proba(x_val_scaled)[:,1]
print(roc_auc_score(y_val, y_val_pred))
y_pred = lgbm.predict_proba(x_test_scaled)[:,1]
my_submission = pd.DataFrame({'id': test.id, 'Response': y_pred})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_lgbm.csv', index=False)
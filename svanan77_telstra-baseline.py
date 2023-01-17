#################################################################
# Libraries
#################################################################

import pandas as pd 
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import seaborn as sns

print("Library Import.... Complete")
#################################################################
# Symlinks
#################################################################

event_type = pd.read_csv("../input/telstra-competition-dataset/event_type.csv")
log_feature = pd.read_csv("../input/telstra-competition-dataset/log_feature.csv")
resource_type = pd.read_csv("../input/telstra-competition-dataset/resource_type.csv")
sample_submission = pd.read_csv("../input/telstra-competition-dataset/sample_submission.csv")
severity_type = pd.read_csv("../input/telstra-competition-dataset/severity_type.csv")
test = pd.read_csv("../input/telstra-competition-dataset/test.csv")
train = pd.read_csv("../input/telstra-competition-dataset/train.csv")

print("Data Loading.... Complete")
#################################################################
# Data Frame Options
#################################################################

#To print dataframe in full without truncation
pd.set_option('expand_frame_repr', False) 	#Print DF in single row
pd.set_option('display.max_columns', None) 	#Print all columns
pd.set_option('display.max_rows', None) 	#Print all rows

print("DF Options Loading.... Complete")
#################################################################
# Helper Function
#################################################################

def mergefiles(dfs):
    countfiles = len(dfs)

    for i in range(countfiles):
        if i == 0:
            dfm = dfs[i]
        else:
            dfm = pd.merge(dfm,dfs[i],on="id")
    
    return dfm

print("Helper function Loading.... Complete")
# Getting dataframe details e.g columns, data types, total entries etc
train.info()
# Viewing top few lines of the dataframe
train.head()
# Reviewing rows
print('Total row entries               : ', len(train.index))
print('Number of unique ids            : ', len(train.id.unique()))
print('Number of unique location       : ', len(train.location.unique()))
print('Number of unique fault_severity : ', len(train.fault_severity.unique()))
# Checking if there are missing values in the data frame
train.isnull().sum()
# Getting dataframe details e.g columns, data types, total entries etc
test.info()
# Viewing top few lines of the dataframe
test.head()
# Reviewing rows
print('Total row entries               : ', len(test.index))
print('Number of unique ids            : ', len(test.id.unique()))
print('Number of unique location       : ', len(test.location.unique()))
# Checking if there are missing values in the data frame
test.isnull().sum()
#Create new column with origin information
train['istrain'] = 1 #Train set
test['istrain'] = 0 #Test set

#Merge dataframes
data = train.append(test, sort=False)
print("Train and Test data merger.... Complete")
# Getting dataframe details e.g columns, data types, total entries etc
data.info()
# Viewing top few lines of the dataframe
data.head()
# Viewing bottom few lines of the dataframe
data.tail()
# Reviewing rows
print('Total row entries               : ', len(data.index))
print('Number of unique ids            : ', len(data.id.unique()))
print('Number of unique location       : ', len(data.location.unique()))
print('Number of unique fault_severity : ', len(data.fault_severity.unique()))
# Checking if there are missing values in the data frame
data.isnull().sum()
#Stripping the string 'location '
data.location = data.location.str.lstrip('location ').astype('int')

data.head()
#Frequency of location
plt.figure(figsize=(22,6))

col='location'
ax=sns.countplot(x = col,
                 data = data,
                 order = data[col].value_counts().index)

plt.xlabel(col)
plt.ylabel('Frequency')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=6)

plt.show()
# Frequence of location
# - Zoom to view top 50
plt.figure(figsize=(22,6))

col='location'
ax=sns.countplot(x = col,
                 data = data,
                 order = data[col].value_counts()[:50,].index)

plt.xlabel(col)
plt.ylabel('Frequency')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=10)

plt.show()
#Frequency of fault_severity
plt.figure(figsize=(12,6))

col='fault_severity'
pd.value_counts(data[col]).plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
#Frequency of istrain
plt.figure(figsize=(12,6))

col='istrain'
pd.value_counts(data[col]).plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
#location vs fault_severity
data.plot.scatter(x='id',
                  y='location',
                  c='fault_severity',
                  colormap='viridis',
                  figsize=(18,12))
# Getting dataframe details e.g columns, data types, total entries etc
event_type.info()
# Viewing top few lines of the dataframe
event_type.head()
# Reviewing rows
print('Total row entries               : ', len(event_type.index))
print('Number of unique ids            : ', len(event_type.id.unique()))
print('Number of unique event_type     : ', len(event_type.event_type.unique()))
# Checking if there are missing values in the data frame
event_type.isnull().sum()
#Removing event_type from column
event_type.event_type= event_type.event_type.str.lstrip('"event_type ')

event_type.head()
#Frequency of event_type
plt.figure(figsize=(18,6))

col='event_type'
ax=sns.countplot(x = col,
                 data = event_type,
                 order = event_type[col].value_counts().index)

plt.xlabel(col)
plt.ylabel('Frequency')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=10)

plt.show()
#One Hot Encoding using Panda
event_type = pd.get_dummies(event_type, columns=['event_type'])

# Viewing top few lines of the dataframe
event_type.head(5)
# Getting dataframe details e.g columns, data types, total entries etc
log_feature.info()
# Viewing top few lines of the dataframe
log_feature.head()
# Reviewing rows
print('Total row entries            : ', len(log_feature.index))
print('Number of unique ids         : ', len(log_feature.id.unique()))
print('Number of unique log_feature : ', len(log_feature.log_feature.unique()))
print('Number of unique volume      : ', len(log_feature.volume.unique()))
# Checking if there are missing values in the data frame
log_feature.isnull().sum()
#Removing log_feature string
log_feature.log_feature = log_feature.log_feature.map(lambda x: x.lstrip('feature '))

log_feature.head()
#Frequency of log_feature
plt.figure(figsize=(25,6))

col='log_feature'
pd.value_counts(log_feature[col]).plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
#Frequency of log_feature
# - Zoom to view the top 50
plt.figure(figsize=(25,6))

col='log_feature'
pd.value_counts(log_feature[col])[:50,].plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
#Frequency of volume
plt.figure(figsize=(25,6))

col='volume'
pd.value_counts(log_feature[col]).plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
#Frequency of volume
# - Zoom to view top 50
plt.figure(figsize=(25,6))

col='volume'
pd.value_counts(log_feature[col])[:50,].plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
# Getting dataframe details e.g columns, data types, total entries etc
resource_type.info()
# Viewing top few lines of the dataframe
resource_type.head()
# Reviewing rows
print('Total row entries               : ', len(resource_type.index))
print('Number of unique ids            : ', len(resource_type.id.unique()))
print('Number of unique resource_type  : ', len(resource_type.resource_type.unique()))
# Checking if there are missing values in the data frame
resource_type.isnull().sum()
#Removing resource_type from column before applying one hot encoding
resource_type.resource_type = resource_type.resource_type.str.lstrip('resource_type ')

resource_type.head()
# Frequency of resource_type
plt.figure(figsize=(12,6))

col='resource_type'
pd.value_counts(resource_type[col]).plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
#One Hot Encoding using Panda
resource_type = pd.get_dummies(resource_type, columns=['resource_type'])

resource_type.head()
# Getting dataframe details e.g columns, data types, total entries etc
severity_type.info()
# Viewing top few lines of the dataframe
severity_type.head()
# Reviewing rows
print('Total row entries               : ', len(severity_type.index))
print('Number of unique ids            : ', len(severity_type.id.unique()))
print('Number of unique severity_type  : ', len(severity_type.severity_type.unique()))
# Checking if there are missing values in the data frame
severity_type.isnull().sum()
#Removing severity_type from column before applying one hot encoding
severity_type.severity_type = severity_type.severity_type.str.lstrip('severity_type ')

severity_type.head()
#Frequency of Severity Type
plt.figure(figsize=(12,6))

col='severity_type'
pd.value_counts(severity_type[col]).plot.bar()

plt.xlabel(col)
plt.ylabel('Frequency')

plt.show()
#One Hot Encoding using Panda
severity_type = pd.get_dummies(severity_type, columns=['severity_type'])

severity_type.head()
# Getting dataframe details e.g columns, data types, total entries etc
sample_submission.info()
# Viewing top few lines of the dataframe
sample_submission.head()
#event_type after one hot encoding is 'sum'med, this is same as 'count' too
event_type = event_type.groupby('id', sort=False).agg(sum).add_prefix('sum_').reset_index()

event_type.head()
len(event_type.index)
#log_feature's 'log_feature' is counted, 'volume' is 'sum'med and 'average'd
log_feature = log_feature.groupby('id', sort=False).agg(count_log_feature=('log_feature','count'), 
                                                        sum_volume=('volume', 'sum'),
                                                        mean_volume=('volume', 'mean')
                                                      ).reset_index()
log_feature.head()
len(log_feature.index)
#resource_type after one hot encoding is 'sum'med, this is same as 'count' too
resource_type = resource_type.groupby('id', sort=False).agg(sum).add_prefix('sum_').reset_index()

resource_type.head()
# Merging dataframes
dfs = [data, log_feature, severity_type, resource_type, event_type] # list of dataframes
result = mergefiles(dfs)

result.head()

len(result.index)
# Checking if there are missing values in the data frame
result.isnull().sum()
#                        (Dataset)
# ┌───────────────────────────────────────────────────────┐  
#  ┌──────────────────────────┬─────────────┐  ┌─────────┐
#  │          Training      │ Validation │  │  Test  │
#  └──────────────────────────┴─────────────┘  └─────────┘
    
# Splitting train data from result datafram by istrain column
train = result[result['istrain'] == 1]
train=train.reset_index(drop=True)
train.head()
# Splitting test data from result datafram by istrain column
test = result[result['istrain'] == 0]
test=test.reset_index(drop=True)
test.head()
#################################################################
# Dataset (Train Test Split)
# - Train (80% of train_new)
# - Validation (20% of train_new)
#################################################################
from sklearn.model_selection import train_test_split

# Selected features - Training data
X = train.drop(columns='fault_severity')

# Prediction target - Training data
y = train['fault_severity']

# Selected features - Test data
x = test.drop(columns='fault_severity')

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

print("Train Validation Split Complete")
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
#################################################################
# Pipeline
#################################################################
pipe_gnb = Pipeline([
    ('scl', None),
    ('pca', None),
    ('clf', GaussianNB())
    ])

#################################################################
# Parameter
#################################################################
parameters_gnb = {
        'clf__priors':[None]
    }

#################################################################
# Grid Search
#################################################################
grid_gnb = GridSearchCV(pipe_gnb,
    param_grid=parameters_gnb,
    scoring='neg_mean_absolute_error',
    cv=5,
    refit=True) 

print("Pipeline Complete")
import time
start_time = time.time()

grid = grid_gnb

print('Performing model optimization...')

print('\nEstimator: GaussianNB')   
    
# Fit grid search   
grid.fit(X_train, y_train)
    
#Calculate the Mean Absolute Error score.
mae = grid.score(X_valid,y_valid)

#################################################################
# Prediction
#################################################################
#Predict using the test data with selected features
y_pred = grid.predict_proba(x)

# Transform numpy array to dataframe
y_pred = pd.DataFrame(y_pred)

# Rearrange dataframe
y_pred.columns = ['predict_0', 'predict_1', 'predict_2']

y_pred.insert(0, 'id', x['id'])

fname = "telstra_baseline_GaussinNB_predict.csv"

# Save to CSV
y_pred.to_csv(fname, index = False, header=True)

print("Best params                       : %s" % grid.best_params_)
print("Best training data MAE score      : %s" % grid.best_score_)    
print("Best validation data MAE score (*): %s" % mae)
print("Modeling time                     : %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
print('y_pred                            :\n %s' % y_pred.head())

#Performing model optimizations...
#Estimator: GaussianNB
#Best params                       : {'clf__priors': None}
#Best training data MAE score      : -0.7249354898893497
#Best validation data MAE score (*): -0.7020988490182803
#Modeling time                     : 00:00:00
#y_pred                            :
#       id  predict_0  predict_1     predict_2
#0  11066   0.999972   0.000028  6.850049e-20
#1  18000   0.001623   0.007001  9.913755e-01
#2  16964   0.998155   0.001845  5.382375e-20
#3   4795   0.000167   0.999833  2.295166e-09
#4   3392   0.109785   0.073884  8.163309e-01
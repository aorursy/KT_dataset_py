import numpy as np 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures

import time
import os
print(os.listdir("../input"))
app_train = pd.read_csv('../input/application_train.csv')
app_train.head()
app_test = pd.read_csv('../input/application_test.csv')
app_test.head()
target = app_train.loc[:, 'TARGET']
columns_feature = app_train.columns.tolist()
columns_feature.remove('TARGET')
features = app_train.loc[:, columns_feature]
features.shape
features = app_test
features.shape
df = pd.DataFrame(app_train['TARGET'].value_counts())
df['repaid'] = ['repaid', 'not repaid']
df.columns=['frequency', 'repaid']
df
sns.set(style="whitegrid")
size=(5, 5)
fig, ax = plt.subplots(figsize=size)
ax.set_title('TARGET distribution')
sns.barplot(x="repaid", y="frequency", data=df, ax=ax)
# calculate null function
def calCompleteness(data):
    row_num = data.shape[0]

    nul = data.isnull().sum()
    nul = pd.DataFrame({'features': nul.index, 'null_number': nul.values})

    comp = []
    for index, row in nul.iterrows():
        temp = row['null_number']
        re = float(row_num - temp) / row_num * 100
        comp.append(re)

    nul['completeness'] = pd.to_numeric(comp, downcast='float')
    nul = nul.sort_values(by=['completeness'], ascending=False)
    nul = nul.reset_index(drop=True)
    return nul

complete = calCompleteness(features)
sns.set(style="whitegrid")
size=(20, 40)
fig, ax = plt.subplots(figsize=size)
ax.set_title('Training Data Completeness')
sns.barplot(x="completeness", y="features", data=complete, ax=ax)
app_train.dtypes.value_counts()
# pd.Series.nunique: Return number of unique elements in the object.
temp = app_train.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
pd.DataFrame({'feature': temp.index, 'Number of Categories': temp.values})
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

# Iterate through the columns
for col in app_train:
    if app_train[col].dtype == 'object':
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # Keep track of how many columns were label encoded
            le_count += 1
            
print('%d columns were label encoded.' % le_count)
# one-hot encoding
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
train_labels = app_train['TARGET']
# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join = 'inner', axis = 1)
# Add the target back in
app_train['TARGET'] = train_labels

print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)
# DAYS_BIRTH analysis
# age description
(app_train['DAYS_BIRTH'] / -365).describe()
# DAYS_EMPLOYED analysis
# get employed years
(app_train['DAYS_EMPLOYED'] / 365).describe()
(app_train['DAYS_EMPLOYED']/356).plot.hist(title = 'Years Employment Histogram');
plt.xlabel('Years Employment');
# Create an anomalous flag column
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243

# Replace the anomalous values with nan
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

app_train['DAYS_EMPLOYED'].plot.hist(title = 'Days Employment Histogram');
plt.xlabel('Days Employment');
# do same thing on the testing data
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

print('There are %d anomalies in the test data out of %d entries' % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test)))
# Find correlations with the target and sort
corr = app_train.corr()
correlations = corr['TARGET'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
# Drop the target from the training data
if 'TARGET' in app_train:
    train = app_train.drop(columns = ['TARGET'])
else:
    train = app_train.copy()
    
# Feature names
features = list(train.columns)

# Copy of the testing data
test = app_test.copy()

# Median imputation of missing values
imputer = Imputer(strategy = 'median')

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Fit on the training data
imputer.fit(train)

# Transform both training and testing data
train = imputer.transform(train)
test = imputer.transform(app_test)

# Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

print('Training data shape: ', train.shape)
print('Testing data shape: ', test.shape)
from sklearn.linear_model import LogisticRegression

# Make the model with the specified regularization parameter
log_reg = LogisticRegression(C = 0.0001, solver='lbfgs')

# Train on the training data
log_reg.fit(train, train_labels)

# Make predictions
# Make sure to select the second column only
log_reg_pred = log_reg.predict_proba(test)[:, 1]
# Submission dataframe
submit = app_test.loc[:, ['SK_ID_CURR']]
submit['TARGET'] = log_reg_pred

submit.head()
# Save the submission to a csv file
submit.to_csv('baseling_submission.csv', index = False)
# Extracting the libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import numpy as np

import matplotlib.pyplot as plt

import xgboost as xgb

import seaborn as sns

import warnings

from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, roc_curve

from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv('/kaggle/input/retailtransactiondata/Retail_Data_Response.csv')

df2 = pd.read_csv('/kaggle/input/retailtransactiondata/Retail_Data_Transactions.csv',parse_dates=['trans_date'])
df1.head()
df2.head()
df1.describe(include='all')
df2.describe(include='all')
print(df2['trans_date'].min())

print(df2['trans_date'].max())
sd = dt.datetime(2015,3,17)

df2['recent']= sd - df2['trans_date']

df2['recent'].astype('timedelta64[D]')

df2['recent']=df2['recent'] / np.timedelta64(1, 'D')

df2.head()
data_rfm = df2.groupby('customer_id').agg({'recent': lambda x:x.min(), # Recency

                                        'customer_id': lambda x: len(x),               # Frequency

                                        'tran_amount': lambda x: x.sum()})          # Monetary Value



data_rfm.rename(columns={'recent': 'recency', 

                         'customer_id': 'frequency', 

                         'tran_amount': 'monetary_value'}, inplace=True)
rfm = data_rfm.reset_index()
rfm.head()
rfm.describe(include='all')
label = df1.groupby('response').agg({'customer_id': lambda x: len(x)})

label.head()
plt.figure(figsize=(5,5))

x=range(2)

plt.bar(x,label['customer_id'])

plt.xticks(label.index)

plt.title('Label Distribution')

plt.xlabel('Convert or Not')

plt.ylabel('total_user')

plt.show()
dataset = pd.merge(df1,rfm)

dataset.head()
# Define the minority data size and indices

minority_class_len = len(dataset[dataset['response'] == 1])

minority_index_list = dataset[dataset['response'] == 1].index

print(minority_index_list)
#Define the majority data size and indices

majority_class_len = len(dataset[dataset['response'] == 0])

majority_index_list = dataset[dataset['response'] == 0].index

print(majority_index_list)
#Perform random undersampling

random_majority = np.random.choice(majority_index_list,

                                   minority_class_len,

                                   replace = False)

under_sample_indexlist = np.concatenate([random_majority,minority_index_list])

under_sample = dataset.loc[under_sample_indexlist]

under_sample.reset_index(drop=True, inplace=True)

under_sample.head()
label_us = under_sample.groupby('response').agg({'customer_id': lambda x: len(x)})

label_us.head()
plt.figure(figsize=(5,5))

x=range(2)

plt.bar(x,label_us['customer_id'])

plt.xticks(label_us.index)

plt.title('Label Distribution')

plt.xlabel('Convert or Not')

plt.ylabel('total_user')

plt.show()
from sklearn.utils import resample
# Separate input features and target

y = dataset.response

X = dataset.drop('response', axis=1)



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)
# concatenate our training data back together

X = pd.concat([X_train, y_train], axis=1)

X.head()
# separate minority and majority classes

not_order = X[X.response==0]

order = X[X.response==1]



# upsample minority

order_upsampled = resample(order,

                          replace=True, # sample with replacement

                          n_samples=len(not_order), # match number in majority class

                          random_state=27) # reproducible results



# combine majority and upsampled minority

upsampled = pd.concat([not_order, order_upsampled])

upsampled.reset_index(drop=True, inplace=True)

upsampled.head()
label_os = upsampled.groupby('response').agg({'customer_id': lambda x: len(x)})

label_os.head()
plt.figure(figsize=(5,5))

x=range(2)

plt.bar(x,label_os['customer_id'])

plt.xticks(label_os.index)

plt.title('Label Distribution')

plt.xlabel('Convert or Not')

plt.ylabel('total_user')

plt.show()
x = under_sample.drop(columns=['response','customer_id'])

y = under_sample['response']

identifier = under_sample['customer_id']



for i in range(0,100):

    skf = StratifiedKFold(n_splits=5, random_state = i, shuffle = True)

        

predicted_y = []

expected_y = []

customer_id = []



for train_index, test_index in skf.split(x, y):

    x_train, x_test = x.loc[train_index], x.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    id_train, id_test = identifier[train_index], identifier[test_index]

    

    xgb_model = xgb.XGBClassifier(objective='binary:logistic').fit(x.loc[train_index], y[train_index])

    predictions = xgb_model.predict(x.loc[test_index])



    predicted_y.extend(predictions)

    expected_y.extend(y_test)

    customer_id.extend(id_test)

    

result = {'id': customer_id,'pred': predicted_y, 'exp': expected_y}    

report = classification_report(expected_y, predicted_y)

print(report)
score = pd.DataFrame(data=result)

score.head()
cf_matrix = confusion_matrix(expected_y,predicted_y)

plt.figure(figsize=(10,9))

group_names = ['TN','FP','FN','TP']

group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cf_matrix, annot = labels, fmt='', cmap='Blues')

plt.show()
x = upsampled.drop(columns=['response','customer_id'])

y = upsampled['response']

identifier = upsampled['customer_id']





for i in range(0,100):

    skf = StratifiedKFold(n_splits=10, random_state = i, shuffle = True)

         

predicted_y = []

expected_y = []

customer_id = []



for train_index, test_index in skf.split(x, y):

    x_train, x_test = x.loc[train_index], x.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    id_train, id_test = identifier[train_index], identifier[test_index]

    

    xgb_model = xgb.XGBClassifier(objective='binary:logistic').fit(x.loc[train_index], y[train_index])

    predictions = xgb_model.predict(x.loc[test_index])



    predicted_y.extend(predictions)

    expected_y.extend(y_test)

    customer_id.extend(id_test)

    

result = {'id': customer_id,'pred': predicted_y, 'exp': expected_y}    

report = classification_report(expected_y, predicted_y)

print(report)
score = pd.DataFrame(data=result)

score.head()
cf_matrix = confusion_matrix(expected_y,predicted_y)

plt.figure(figsize=(10,9))

group_names = ['TN','FP','FN','TP']

group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cf_matrix, annot = labels, fmt='', cmap='Blues')

plt.show()
x = upsampled.drop(columns=['response','customer_id'])

y = upsampled['response']

identifier = upsampled['customer_id']



for i in range(0,100):

    skf = StratifiedKFold(n_splits=10, random_state = i, shuffle = True)

        

predicted_y = []

expected_y = []

customer_id = []



for train_index, test_index in skf.split(x, y):

    x_train, x_test = x.loc[train_index], x.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    id_train, id_test = identifier[train_index], identifier[test_index]

    

    xgb_model = xgb.XGBClassifier(objective='binary:logistic').fit(x.loc[train_index], y[train_index])

    predictions = xgb_model.predict_proba(x.loc[test_index])[:,1]

    

    predicted_y.extend(predictions)

    expected_y.extend(y_test)

    customer_id.extend(id_test)

    

    

result = {'id': customer_id,'pred': predicted_y, 'exp': expected_y} 

prob_score = pd.DataFrame(data=result)

prob_score.head()
prob_score.to_csv('xgboost_propensity_score.csv',index=False)
x = upsampled.drop(columns=['response','customer_id'])

y = upsampled['response']



logreg = LogisticRegression(solver='liblinear', penalty='l1', C=0.1, class_weight='balanced')



for i in range(0,100):

    skf = StratifiedKFold(n_splits=10, random_state = i, shuffle = True)

        

predicted_y = []

expected_y = []



for train_index, test_index in skf.split(x, y):

    x_train, x_test = x.loc[train_index], x.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    logreg_model = logreg.fit(x.loc[train_index], y[train_index])

    predictions = logreg_model.predict(x.loc[test_index])



    predicted_y.extend(predictions)



    expected_y.extend(y_test)

    

report = classification_report(expected_y, predicted_y)

print(report)
cf_matrix = confusion_matrix(expected_y,predicted_y)

plt.figure(figsize=(10,9))

group_names = ['TN','FP','FN','TP']

group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cf_matrix, annot = labels, fmt='', cmap='Blues')

plt.show()
x = under_sample.drop(columns=['response','customer_id'])

y = under_sample['response']



logreg = LogisticRegression(solver='liblinear', penalty='l1', C=0.1, class_weight='balanced')



for i in range(0,100):

    skf = StratifiedKFold(n_splits=10, random_state = i, shuffle = True)

        

predicted_y = []

expected_y = []



for train_index, test_index in skf.split(x, y):

    x_train, x_test = x.loc[train_index], x.loc[test_index]

    y_train, y_test = y[train_index], y[test_index]

    

    logreg_model = logreg.fit(x.loc[train_index], y[train_index])

    predictions = logreg_model.predict(x.loc[test_index])



    predicted_y.extend(predictions)



    expected_y.extend(y_test)

    

report = classification_report(expected_y, predicted_y)

print(report)
cf_matrix = confusion_matrix(expected_y,predicted_y)

plt.figure(figsize=(10,9))

group_names = ['TN','FP','FN','TP']

group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cf_matrix, annot = labels, fmt='', cmap='Blues')

plt.show()
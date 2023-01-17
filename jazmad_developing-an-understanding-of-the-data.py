# Import packages

import numpy as np

import matplotlib.pyplot as plt

#%matplotlib inline

import pandas as pd

import os

import time

import xgboost as xgb

from sklearn.model_selection import cross_val_score

import ipywidgets as widgets

from IPython.display import display
t_Start = time.time()
t1_Start = time.time()
# Created dataframes from input files

train_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

test_id = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

train_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

test_trans = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

sample_sub = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')



print('Data imported')
t1_End = time.time()

if (t1_End - t1_Start) < 60:

    print('Step 1 completed - Time taken %3.1f seconds' % (t1_End - t1_Start))

else:

    print('Step 1 completed - Time taken %i minutes and %i seconds' % ((t1_End - t1_Start)//60,(t1_End - t1_Start)%60))
t2_Start = time.time()
avg_fraud_rate = train_trans.isFraud.sum()/train_trans.shape[0]
# Merge the ID data and the transaction data



train_merged = pd.merge(train_trans, train_id, how='outer', on='TransactionID')
w = widgets.Dropdown(options=train_merged.columns.drop(['TransactionID','isFraud']), value='card6', description='column:', disabled=False)
# Function to create variables based on selected column



def Display_Features(column):

    ''' (str) --> None

    

    For the selected column, displays various features

    '''

    # Data type

    data_type = train_merged[column].dtype



    # Proportion of missing values

    missing_proportion = train_merged[column].isna().sum()/train_merged[column].count()

    

    # Dataframe of most common values and total other values 

    top_9 = train_merged.groupby(column).count()['TransactionID'].sort_values(ascending=False).head(9)

    top_10 = top_9.copy()

    top_10.loc['other'] = train_merged.groupby(column).count()['TransactionID'].sum()-top_9.sum()

    

    print('The data type is %s' % str(data_type))

    print('%3.1f%% of the data is missing' % (missing_proportion*100))

    display(top_10.to_frame())

# Bar chart of frequencies



def Display_Chart(column):

    ''' (str) --> None

    

    For the selected column, displays a plot of frequencies and fraud rate

    '''

    fig, ax = plt.subplots(1, 1, figsize=(8, 5));



    # bar chart of frequencies

    train_merged.groupby(column).count()['TransactionID'].plot.bar(color='blue',alpha=0.5,ax=ax);

    bx = ax.twinx()

    ax.set_title('Frequency of values (fraud % overlaid)');

    ax.set_xlabel('Value');

    ax.set_ylabel('Frequency');

    ax.set_xticks(range(len(train_merged.groupby(column).count())))

    ax.set_xticklabels(train_merged.groupby(column).count()['TransactionID'].index)

    # line chart of fraud rate

    (1-train_merged[train_merged['isFraud']==0].groupby(column).count()/train_merged.groupby(column).count())['TransactionID'].plot.line(ax=bx,color='red')

    bx.set_ylabel('Fraud %');

    

    plt.show()

    
selection = widgets.Output()



def Widger_Event_Handler(change):

    ''' None --> None

    Function to automatically display dashboard when widget changed

    '''

    selection.clear_output()

    with selection:

        Display_Features(change.new)

        Display_Chart(change.new)

    

w.observe(Widger_Event_Handler, names='value')



display(w)
display(selection)
t2_End = time.time()

if (t2_End - t2_Start) < 60:

    print('Step 2 completed - Time taken %3.1f seconds' % (t2_End - t2_Start))

else:

    print('Step 2 completed - Time taken %i minutes and %i seconds' % ((t2_End - t2_Start)//60,(t2_End - t2_Start)%60))
t3_Start = time.time()
# Set up target variable



train_trans = train_trans.set_index('TransactionID')

y_train = train_trans.isFraud

# Define a function to manipulate the data for input into the model

# At this stage, just cut the data down to fully populated, numeric columns



def DataForInput(trans_data,ID_data):

    ''' (df, df) --> df

    

    Takes the transaction and ID dataframes and combines them to create a single dataframe suitable for input into the model.

    

    Set up as a function so can be repeated for training and test dataset

    '''

    

    return trans_data['TransactionAmt'] 
# Set up the training data ready for model training



X_train = DataForInput(train_trans,train_id)
# Fit a model



#xgboost = xgb.XGBClassifier().fit(X_train,y_train)
# Work out some performance statistics using cross-validation

'''

xgb_accuracy = cross_val_score(xgboost, X_train, y_train, cv=5).mean()

xgb_auc = cross_val_score(xgboost, X_train, y_train, scoring='roc_auc', cv=5).mean()



print('The CV model accuracy score is %3.1f%%' % (xgb_accuracy*100))

print('The CV model AUC score is %3.5f' % (xgb_auc))

'''
t3_End = time.time()

if (t3_End - t3_Start) < 60:

    print('Step 3 completed - Time taken %3.1f seconds' % (t3_End - t3_Start))

else:

    print('Step 3 completed - Time taken %i minutes and %i seconds' % ((t3_End - t3_Start)//60,(t3_End - t3_Start)%60))
t4_Start = time.time()
# Create predictions



test_trans = test_trans.set_index('TransactionID')

X_test = DataForInput(test_trans,test_id)

#X_test.preds = xgboost.predict_proba(X_test)

X_test.preds = 0.5

sample_sub.isFraud = 1 - X_test.preds

sample_sub.to_csv('submission.csv',index=False)
plt.figure(figsize=(12, 5))

plt.hist(sample_sub.isFraud,bins=100);

plt.title('Distribution of prediction probabilities in submission');

plt.xlabel('Probability');

plt.ylabel('Count');
mean = sample_sub.isFraud.mean()



print('The predicted fraud rate is %3.2f%%' % (mean*100))
t4_End = time.time()

if (t4_End - t4_Start) < 60:

    print('Step 4 completed - Time taken %3.1f seconds' % (t4_End - t4_Start))

else:

    print('Step 4 completed - Time taken %i minutes and %i seconds' % ((t4_End - t4_Start)//60,(t4_End - t4_Start)%60))
t_End = time.time()

print('Notebook finished - Total run time = %i minutes and %i seconds' % ((t_End - t_Start)//60,(t_End - t_Start)%60))
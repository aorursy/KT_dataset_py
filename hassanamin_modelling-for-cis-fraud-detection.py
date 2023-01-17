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
def load_data():

    # Loading

    df_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

    df_test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')



    df_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

    df_test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

    # Merging

    df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True)

    df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True)

    

    return df_train,df_test

df_train, df_test = load_data()

print(df_train.shape)

print(df_test.shape)
#df_train = reduce_mem_usage(df_train)

#df_test = reduce_mem_usage(df_test)
# resumetable(df_trans)[:25]
print(f'Train dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns.')

print(f'Test dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns.')
import matplotlib.pyplot as plt

plt.hist(df_train['isFraud'])
from sklearn.model_selection import train_test_split

#sel_features=['isFraud','TransactionDT','TransactionAmt']

#features = sel_features+num_id+sel_cards

# Selecting numeric columns in df_train

def stratified_sampling(input_df):

    print(input_df.select_dtypes('number').columns)

    sel_train = input_df.select_dtypes('number').columns.values

    print(type(sel_train))



    train = input_df[sel_train]

    print(train.describe())

    y = train['isFraud']

    X = train

    X.drop(["isFraud"],axis=1,inplace=True)

    ### Train-test split with Stratification

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,  test_size=0.25)

    return X_train, X_test, y_train, y_test

def numeric_cols(input_df):

    # Selecting numeric columns in df_train

    print(input_df.select_dtypes('number').columns)

    sel_train = input_df.select_dtypes('number').columns.values

    print(type(sel_train))



    train = input_df[sel_train]

    print(train.describe())

    return train
# Returns list of categorical columns, and part of dataset with only categorical columns

def categorical_cols(input_df):

    # Selecting numeric columns in df_train

    print(input_df.select_dtypes('object').columns)

    sel_train = input_df.select_dtypes('object').columns.values

    #print(type(sel_train))



    train = input_df[sel_train]

    #print(train.describe())

    return sel_train, train
#Label encoding selected categorical columns, while leaving other columns as it is

from sklearn.preprocessing import LabelEncoder

def label_encoding(sel_cat,inpX):

    for col in sel_cat:

        if col in inpX.columns:

            le = LabelEncoder()

            le.fit(list(inpX[col].astype(str).values))

            inpX[col] = le.transform(list(inpX[col].astype(str).values))

    return inpX

from sklearn.model_selection import train_test_split



#features = sel_features+num_id+sel_cards

#train = df_train[features]

def balanced_sampling(input_df): 

    

    train = numeric_cols(input_df)

    y= train['isFraud']

    # Selecting fraud and no fraud  

    X_fraud= train[train.isFraud==1]

    X_nofraud= train[train.isFraud==0]

    total_fraud = X_fraud.shape

    print(total_fraud,total_fraud[0])

    

    scale_factor = 3

    X_nofraud1=X_nofraud.sample(scale_factor * total_fraud[0])

    

    X=pd.concat([X_fraud,X_nofraud1], ignore_index=True)

    

    y= X['isFraud']

    print(X.shape)

    print(X.sample(10))



    #dropping isFraud column from X

    X.drop(["isFraud"],axis=1,inplace=True)

    

    ### Train-test split with Stratification

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,  test_size=0.25)

    return X_train, X_test, y_train, y_test

def preprocess(inp):

# Filling 0.0 in place of NaN

    inp.fillna(0.0, inplace=True)

    inp.sample(10)

    return inp 
from sklearn.preprocessing import StandardScaler

def scaling(unscaled_data):

    #unscaled_data.reset_index()

    ss = StandardScaler()

    #preprocessing to remove NaN's

    processed_data=preprocess(unscaled_data)

    #scaling

    scaled_data = ss.fit_transform(processed_data)

    #print('Unscaled Data:\n',X)

    #print("Scaled Data :\n",scaled_data)

    return scaled_data

#Import Random Forest Model

from sklearn.ensemble import RandomForestClassifier

def randomforest(inpX,inpy):

    #Create a Gaussian Classifier

    clf=RandomForestClassifier(n_estimators=500)



    #Train the model using the training sets y_pred=clf.predict(X_test)

    clf.fit(X_train,y_train)

    return clf

# Step 1 : Load Data

df_train,df_test = load_data()

# Step 2 : Balanced sampling with numeric columns

X_train, X_test, y_train, y_test = balanced_sampling(df_train)
scaledX_train = scaling(X_train)

scaledX = pd.DataFrame(scaledX_train)

scaledX.sample(10)
clf = randomforest(scaledX,y_train)

scaledX_test = scaling(X_test)

scaledX_test = pd.DataFrame(scaledX_test)

y_pred=clf.predict(scaledX_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Setup test data with numeric cols only

test = numeric_cols(df_test)

scaledX_test = scaling(test)
# Use df_test with selected columns for final submission

y_preds = clf.predict_proba(scaledX_test)[:,1] 

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

sample_submission['isFraud'] = y_preds

sample_submission.to_csv('submission_scaled.csv')
# Step 2 : Balanced sampling with numeric columns

X_train, X_test, y_train, y_test = balanced_sampling(df_train)
processed_X =  preprocess(X_train)
clf = randomforest(processed_X,y_train)
processed_testX=preprocess(X_test)
y_pred=clf.predict(processed_testX)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



def eval2(y_test,y_pred):

    # Model Accuracy, how often is the classifier correct?

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    return 0
eval2(y_test,y_pred)
def sub2(inp,clf):

    # Setup test data with numeric cols only

    test = numeric_cols(inp)

    processed_testX=preprocess(test)

    #test = df_test[sel_train]

    # Use df_test with selected columns for final submission

    y_preds = clf.predict_proba(processed_testX)[:,1] 

    sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

    sample_submission['isFraud'] = y_preds

    sample_submission.to_csv('RandomForest_model.csv')

    return 0

sub2(df_test, clf)
# Step 1 : Load Data

df_train,df_test = load_data()
sel_cat,X = categorical_cols(df_train)

X = pd.DataFrame(X)

X.sample(10)
# Select Categorical Columns

sel_cat,X = categorical_cols(df_train)

sel_cat2,XX = categorical_cols(df_test)

df_test[sel_cat2].head(10)
# Encode Catorical Columns in training dataset

df_train = label_encoding(sel_cat,df_train)

df_test = label_encoding(sel_cat2,df_test)

df_test.head(10)
df_train.sample(100)
# pre-process train and test datasets to remove NaNs

processed_trainX =  preprocess(df_train)

processed_testX = preprocess(df_test)
# Balanced sampling with train-test split

X_train, X_test, y_train, y_test = balanced_sampling(processed_trainX)
# classification

clf = randomforest(X_train,y_train)
# prediction

y_pred=clf.predict(X_test)
from sklearn import metrics

def eval2(y_test,y_pred):

    # Model Accuracy, how often is the classifier correct?

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    return 0
eval2(y_test,y_pred)
def sub3(inpt,clf):

    y_preds = clf.predict_proba(processed_testX)[:,1] 

    sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

    sample_submission['isFraud'] = y_preds

    sample_submission.to_csv('RandomForest_3x.csv')

    return 0

sub3(processed_testX,clf)

#df_test.head(10)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

def performance_analysis(y_test,y_pred):

    results = confusion_matrix(y_test, y_pred) 

    print('Confusion Matrix :')

    print(results) 

    print('Accuracy Score :',accuracy_score(y_test, y_pred))

    print ('Report : ')

    print (classification_report(y_test, y_pred))

    return



performance_analysis(y_test,y_pred)
!kaggle competitions submit -c ieee-fraud-detection -f RandomForest_3x.csv -m "Message"
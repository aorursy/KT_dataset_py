# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/santander-customer-transaction-prediction/sample_submission.csv")

test = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

train = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
def load_data():

    df_train = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv',index_col='ID_code')

    df_test = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv', index_col='ID_code')



    

    return df_train,df_test

df_train,df_test = load_data()

print(f'Train dataset has {df_train.shape[0]} rows and {df_train.shape[1]} columns.')

print(f'Test dataset has {df_test.shape[0]} rows and {df_test.shape[1]} columns.')
df_train.head(10)
df_train.describe()
plt.hist(df_train['target'])
#Label encoding selected categorical columns, while leaving other columns as it is

from sklearn import preprocessing



def label_encoding(sel_cat,inpX):

    for col in sel_cat:

        if col in inpX.columns:

            le = preprocessing.LabelEncoder()

            le.fit(list(inpX[col].astype(str).values))

            inpX[col] = le.transform(list(inpX[col].astype(str).values))

    return inpX

# Returns list of categorical columns, and part of dataset with only categorical columns

def categorical_cols(input_df):

    # Selecting numeric columns in df_train

    print(input_df.select_dtypes('object').columns)

    sel_train = input_df.select_dtypes('object').columns.values

    #print(type(sel_train))



    train = input_df[sel_train]

    #print(train.describe())

    return sel_train, train
from sklearn.model_selection import train_test_split



#features = sel_features+num_id+sel_cards

#train = df_train[features]

def balanced_sampling(input_df): 

    

    train = numeric_cols(input_df)

    y= train['target']

    # Selecting target 1 and target 0  

    X_target = train[train.target==1]

    X_notarget= train[train.target==0]

    total_target = X_target.shape

    print("Target Size : ",total_target[1],total_target[0])

    scale_factor = 2

    X_notarget1=X_notarget.sample(scale_factor*total_target[0])

    X=pd.concat([X_target,X_notarget1], ignore_index=True)

    y= X['target']

    print(X.shape)

    print(X.sample(10))



    #dropping target column from X

    X.drop(["target"],axis=1,inplace=True)

    

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

    clf.fit(inpX,inpy)

    return clf

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

def randomforest2(inpX,inpy):

    sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))

    sel.fit(inpX, inpy)

    sel.get_support()

    selected_feat= inpX.columns[(sel.get_support())]

    len(selected_feat)

    return sel, selected_feat
# Step 1 : Load Data

df_train,df_test = load_data()
# Select Categorical Columns

sel_cat,X = categorical_cols(df_train)

df_train[sel_cat].head(10)
# pre-process train and test datasets to remove NaNs

processed_trainX =  preprocess(df_train)

processed_testX = preprocess(df_test)

processed_trainX.sample(100)

processed_testX.sample(100)
# Balanced sampling with train-test split

X_train, X_test, y_train, y_test = balanced_sampling(processed_trainX)
plt.hist(y_train)
# Step 6 : Traing part of classification

clf = randomforest(X_train,y_train)
# prediction

y_pred=clf.predict(X_test)
from sklearn import metrics

def eval2(y_test,y_pred):

    # Model Accuracy, how often is the classifier correct?

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    return 0
eval2(y_test,y_pred)
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
def sub3(inpt,clf):

    # Use df_test with selected columns for final submission

    y_preds = clf.predict_proba(inpt)[:,1] 

    sample_submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv', index_col='ID_code')

    sample_submission['target'] = y_preds

    sample_submission.to_csv('santander2.csv')

    return 0

processed_testX.sample(100)

sub3(processed_testX,clf)
# Step 1 : Load Data

df_train,df_test = load_data()
# Select Categorical Columns

sel_cat,X = categorical_cols(df_train)

df_train[sel_cat].head(10)
# pre-process train and test datasets to remove NaNs

processed_trainX =  preprocess(df_train)

processed_testX = preprocess(df_test)

processed_trainX.sample(100)

processed_testX.sample(100)
# Balanced sampling with train-test split

X_train, X_test, y_train, y_test = balanced_sampling(processed_trainX)
plt.hist(y_train)
sfm, sel_feat = randomforest2(X_train,y_train)
print("Selected Features : \n",sel_feat)
print("Number of features : ", len(sel_feat))
# Create a random forest classifier

clf2 = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)



# Train the classifier

clf2.fit(X_train, y_train)



# Print the name and gini importance of each feature

for feature in zip(sel_feat, clf2.feature_importances_):

    print(feature)
# Transform the data to create a new dataset containing only the most important features

# Note: We have to apply the transform to both the training X and test X data.

X_important_train = sfm.transform(X_train)

X_important_test = sfm.transform(X_test)
X_important_train.shape
X_important_test.shape
clf3 = randomforest(X_important_train,y_train)
# prediction

y_pred=clf3.predict(X_important_test)
from sklearn import metrics

def eval2(y_test,y_pred):

    # Model Accuracy, how often is the classifier correct?

    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    return 
eval2(y_test,y_pred)
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
def sub3(inpt,clf):

    # Use df_test with selected columns for final submission

    y_preds = clf.predict_proba(inpt)[:,1] 

    sample_submission = pd.read_csv('../input/santander-customer-transaction-prediction/sample_submission.csv', index_col='ID_code')

    sample_submission['target'] = y_preds

    sample_submission.to_csv('santander2_1.csv')

    return 0

processed_testX.sample(100)

X_important_df_test = sfm.transform(df_test)

sub3(X_important_df_test,clf3)
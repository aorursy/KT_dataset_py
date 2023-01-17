### IMPORTING REQUIRED PACKAGES



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



# machine learning modules

import sklearn

print(sklearn.__version__)

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn import metrics

from sklearn.metrics import roc_auc_score
#### LOADING DATA ####

### TRAIN DATA

train_data = pd.read_csv("../input/train.csv", na_values='-1')

                        

## Filling the missing data NAN with median of the column

train_data_nato_median = pd.DataFrame()

for column in train_data.columns:

    train_data_nato_median[column] = train_data[column].fillna(train_data[column].median())



train_data = train_data_nato_median.copy()



### TEST DATA

test_data = pd.read_csv("../input/test.csv", na_values='-1')

## Filling the missing data NAN with mean of the column

test_data_nato_median = pd.DataFrame()

for column in test_data.columns:

    test_data_nato_median[column] = test_data[column].fillna(test_data[column].median())

    

test_data = test_data_nato_median.copy()

test_data_id = test_data.pop('id')
## Identifying Categorical data

column_names = train_data.columns

categorical_column = column_names[column_names.str[10] == 'c']



## Changing categorical columns to category data type

def int_to_categorical(data):

    """ 

    changing columns to catgorical data type

    """

    for column in categorical_column:

        data[column] =  data[column].astype('category')

## Creating list of train and test data and converting columns of interest to categorical type

datas = [train_data,test_data]



for data in datas:

    int_to_categorical(data)



print(test_data.dtypes)


## Decribing categorical variables

# def decribe_Categorical(x):

#     """ 

#     Function to decribe Categorical data

#     """

#     from IPython.display import display, HTML

#     display(HTML(x[x.columns[x.dtypes =="category"]].describe().to_html))



# decribe_Categorical(train_data)


### FUNCTION TO CREATE DUMMIES COLUMNS FOR CATEGORICAL VARIABLES

def creating_dummies(data):

    """creating dummies columns categorical varibles

    """

    for column in categorical_column:

        dummies = pd.get_dummies(data[column],prefix=column)

        data = pd.concat([data,dummies],axis =1)

        ## dropping the original columns ##

        data.drop([column],axis=1,inplace= True)


### CREATING DUMMIES FOR CATEGORICAL VARIABLES  

for column in categorical_column:

        dummies = pd.get_dummies(train_data[column],prefix=column)

        train_data = pd.concat([train_data,dummies],axis =1)

        train_data.drop([column],axis=1,inplace= True)





for column in categorical_column:

        dummies = pd.get_dummies(test_data[column],prefix=column)

        test_data = pd.concat([test_data,dummies],axis =1)

        test_data.drop([column],axis=1,inplace= True)



print(train_data.shape)

print(test_data.shape)
#Define covariates in X and dependent variable in y

X = train_data.iloc[:,2:] ## FEATURE DATA

y= train_data.target ### LABEL DATA



### CHECKING DIMENSIONS

print(X.shape)

print(y.shape)
#### SPLITTING DATA INTO TRAIN AND TEST SETS

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=13)





### RANDOM FOREST CLASSIFIER



"""

number of estimators: 200

out of bagging set to True

N_jobs: Use all the available cores= -1

min_sample_leaf: minimum number of samples required to be at a leaf node

"""

RF_model_cat= RandomForestClassifier(200,oob_score=True,random_state=13,

                                     n_jobs = -1, min_samples_leaf = 100)
### FITTING RANDOM MODEL 

RF_model_cat.fit(X_train, y_train)



#Obtain class predictions

y_pred_RF_prob = RF_model_cat.predict_proba(X_test)

print('Predicted probabilities: \n', y_pred_RF_prob)



#Obtain probability predictions

y_pred_RF_class = RF_model_cat.predict(X_test)

print('Predicted classes: \n', y_pred_RF_class)



print('RF Score: ', metrics.accuracy_score(y_test, y_pred_RF_class))



## CONFUSION MATRIX

RF_cm=metrics.confusion_matrix(y_test,y_pred_RF_class)

print(RF_cm)
#### Predicition on test data ####

y_pred_RF_prob = RF_model_cat.predict_proba(test_data)

pred_values= pd.DataFrame(y_pred_RF_prob)



submission_simple_RF= pd.DataFrame()

submission_simple_RF['id'] = test_data_id



submission_simple_RF['target'] = pd.DataFrame(pred_values.iloc[:,1])

submission_simple_RF = submission_simple_RF.set_index('id')



submission_simple_RF.columns

submission_simple_RF.head()

## Write to CSV

#submission_simple_RF.to_csv("Simple Random Forest.csv")
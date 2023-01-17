# Importing the necessary libraries

import pandas as pd

import numpy as np

import seaborn as sbn

import matplotlib.pyplot as plt

# Read the data

pharma_data = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Training_set_begs.csv')
# Observing the first 5 rows of the dataframe

pharma_data.head()
# Heatmap of Correlation matrix consisting of correlation coefficients between the predictors

plt.figure(figsize=(16,10))

sbn.set(font_scale=1.1)

sbn.heatmap(pharma_data.corr(),annot=True,cmap="RdYlGn")

plt.show()
# Information on different columns of the dataset

pharma_data.info()
# Preprocessing for nominal categorical data (one hot encoding)

pharma_data_cat_conv=pd.get_dummies(pharma_data, columns=["Treated_with_drugs","Patient_Smoker","Patient_Rural_Urban","Patient_mental_condition"])

pharma_data_cat_conv.info()
# Preprocessing for categorical data which are already 1 hot coded (replacing missing values by the most frequent value)

pharma_data_cat_conv['A'].fillna(pharma_data_cat_conv['A'].mode()[0],inplace=True)

pharma_data_cat_conv['B'].fillna(pharma_data_cat_conv['B'].mode()[0],inplace=True)

pharma_data_cat_conv['C'].fillna(pharma_data_cat_conv['C'].mode()[0],inplace=True)

pharma_data_cat_conv['D'].fillna(pharma_data_cat_conv['D'].mode()[0],inplace=True)

pharma_data_cat_conv['E'].fillna(pharma_data_cat_conv['E'].mode()[0],inplace=True)

pharma_data_cat_conv['F'].fillna(pharma_data_cat_conv['F'].mode()[0],inplace=True)

pharma_data_cat_conv['Z'].fillna(pharma_data_cat_conv['Z'].mode()[0],inplace=True)





# Check whether any NaN exists 

pharma_data_cat_conv.isna().sum()
# Separate targets from predictors

y = pharma_data_cat_conv[['Survived_1_year']]  # target variable 

X = pharma_data_cat_conv.drop(['Survived_1_year'], axis=1)  # input variables
# Dropping the ids from the list of predictors as they don't have any impact on the target variable

X=X.drop(['Patient_ID','ID_Patient_Care_Situation'], axis=1)

# Dropping 'Number_of_prev_cond' from the list of predictors as this predictor is the sum of the values in A, B, C, D, E, F, Z. 

X=X.drop(['Number_of_prev_cond'], axis=1)

X.head()
# Normalizing the predictors

from sklearn.preprocessing import MinMaxScaler



# Fit scaler on the input variables

norm = MinMaxScaler().fit(X)



# Transform the input variables

X_norm = pd.DataFrame(norm.transform(X))



X_norm.columns=X.columns
# Break off validation data from training data

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val =train_test_split(X_norm, y, 

                                                    test_size=0.2, 

                                                    random_state=1)
X_train.head()


from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV



# Define initial model. Specify a number for seed to ensure same results each run

clf_lr = XGBClassifier(learning_rate =0.1,

n_estimators=500,

max_depth=5,

min_child_weight=1,

gamma=0,

subsample=0.8,

colsample_bytree=0.8,

objective= 'binary:logistic',

nthread=4,

scale_pos_weight=1,

seed=1)



# Define range of the parameters for optimization using RandomizedSearchCV

optimization_dict = {"boosting_type": ['gbdt','dart'],

"n_estimators"     : [200,500,800,1100,1400,1700,2000],

"learning_rate"    : list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)) ,

"max_depth"        : [3,5,7,9,11,13,15,17,19,21,23],

'silent': [False],

'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

'colsample_bylevel': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],

'min_child_weight': [0.5, 1.0, 3.0, 5.0, 7.0, 10.0],

'gamma': [0, 0.25, 0.5, 1.0],

'reg_lambda': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],

'reg_alpha': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]}





# Define RandomizedSearchCV

model = RandomizedSearchCV(clf_lr, optimization_dict, 

                     scoring='roc_auc', n_iter=10, verbose=1,n_jobs=4,iid=False, cv=2)



# Fit RandomizedSearchCV model to training data

lr_baseline_model = model.fit(X_train,y_train)



lr_baseline_model.best_params_, lr_baseline_model.best_score_
# Define model with optimized parameters

model = XGBClassifier(**lr_baseline_model.best_params_,

 objective= 'binary:logistic',

 nthread=4,

 scale_pos_weight=1,

 random_state=1)



# Fit optimized model

lr_baseline_model_op = model.fit(X_train,y_train)



# Predict validation set targets

y_val_pred = lr_baseline_model_op.predict(X_val)



# Predict training set targets

y_train_pred=lr_baseline_model_op.predict(X_train)



# Calculating evaluation metrics for train data and validation data

from sklearn.metrics import accuracy_score,f1_score

ac_score = accuracy_score(y_val, y_val_pred)

f_score = f1_score(y_val, y_val_pred)

ac_score_train = accuracy_score(y_train,y_train_pred)

f_score_train = f1_score(y_train,y_train_pred)





print("Validation accuracy score:", ac_score)

print("Validation F1 Score:", f_score)

print("Train accuracy score:", ac_score_train)

print("Train F1 Score:", f_score_train)
# Read test data

test_new = pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/pharma_data/Testing_set_begs.csv')

# Check NaN values in Test data

test_new.isna().sum()
# Preprocessing for nominal categorical data (one hot encoding)

test_new=pd.get_dummies(test_new, columns=["Treated_with_drugs","Patient_Smoker","Patient_Rural_Urban","Patient_mental_condition"])

test_new.info()
# test data does not have 'Cannot say' in 'Patient_Smoker' column. Therefore, during the one hot encoding of this column for test data, 'Patient_Smoker_Cannot say' column is not created while it exists for train data. Hence, this column is created for test data with all values as 0

test_new.loc[:,'Patient_Smoker_Cannot say']=0
# Re-aaranging the test data columns to match with train data columns

test_new = test_new[['Diagnosed_Condition', 'Patient_Age', 'Patient_Body_Mass_Index', 'A',

       'B', 'C', 'D', 'E', 'F', 'Z', 'Number_of_prev_cond',

       'Treated_with_drugs_DX1 ', 'Treated_with_drugs_DX1 DX2 ',

       'Treated_with_drugs_DX1 DX2 DX3 ',

       'Treated_with_drugs_DX1 DX2 DX3 DX4 ',

       'Treated_with_drugs_DX1 DX2 DX3 DX4 DX5 ',

       'Treated_with_drugs_DX1 DX2 DX3 DX5 ',

       'Treated_with_drugs_DX1 DX2 DX4 ',

       'Treated_with_drugs_DX1 DX2 DX4 DX5 ',

       'Treated_with_drugs_DX1 DX2 DX5 ', 'Treated_with_drugs_DX1 DX3 ',

       'Treated_with_drugs_DX1 DX3 DX4 ',

       'Treated_with_drugs_DX1 DX3 DX4 DX5 ',

       'Treated_with_drugs_DX1 DX3 DX5 ', 'Treated_with_drugs_DX1 DX4 ',

       'Treated_with_drugs_DX1 DX4 DX5 ', 'Treated_with_drugs_DX1 DX5 ',

       'Treated_with_drugs_DX2 ', 'Treated_with_drugs_DX2 DX3 ',

       'Treated_with_drugs_DX2 DX3 DX4 ',

       'Treated_with_drugs_DX2 DX3 DX4 DX5 ',

       'Treated_with_drugs_DX2 DX3 DX5 ', 'Treated_with_drugs_DX2 DX4 ',

       'Treated_with_drugs_DX2 DX4 DX5 ', 'Treated_with_drugs_DX2 DX5 ',

       'Treated_with_drugs_DX3 ', 'Treated_with_drugs_DX3 DX4 ',

       'Treated_with_drugs_DX3 DX4 DX5 ', 'Treated_with_drugs_DX3 DX5 ',

       'Treated_with_drugs_DX4 ', 'Treated_with_drugs_DX4 DX5 ',

       'Treated_with_drugs_DX5 ', 'Treated_with_drugs_DX6','Patient_Smoker_Cannot say',

       'Patient_Smoker_NO', 'Patient_Smoker_YES', 'Patient_Rural_Urban_RURAL',

       'Patient_Rural_Urban_URBAN', 'Patient_mental_condition_Stable',

       ]]
# Dropping 'Number_of_prev_cond' from the list of predictors in test data as this predictor is the sum of the values in A, B, C, D, E, F, Z and therefore not needed

test_new.drop('Number_of_prev_cond',axis=1,inplace=True)
# Normalize the predictors in test data 

test_new_norm = pd.DataFrame(norm.transform(test_new),columns=test_new.columns)


# Predicting the targets for test data

preds_test= lr_baseline_model_op.predict(test_new_norm)





output = pd.DataFrame({'Id':test_new_norm.index,

                       'Survived_1_year': preds_test})

output.to_csv('submission.csv', index=False)



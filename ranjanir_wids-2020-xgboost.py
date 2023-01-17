# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 



# machine learning

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler, normalize
import warnings

warnings.filterwarnings('ignore')
#Load the training and test dataset

train_df = pd.read_csv('/kaggle/input/widsdatathon2020/training_v2.csv')

test_df = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv')
#Know the features of the dataset

print(train_df.info())

print(train_df.shape)
# Train missing values (in percent)

train_missing = (train_df.isnull().sum() / len(train_df)).sort_values(ascending = False)

train_missing.head()

train_missing = train_missing.index[train_missing > 0.75]

print('There are %d columns with more than 75%% missing values' % len(train_missing))

print('The missing columns are %s' % train_missing)

df_train = train_df.drop(columns = train_missing)

df_test = test_df.drop(columns = train_missing)

df_train.shape

df_test.shape
columns_to_drop = train_df[['patient_id', 'hospital_id','icu_id','readmission_status','hospital_death']]

target = train_df['hospital_death']

df_train = df_train.drop(columns = columns_to_drop)

df_test = df_test.drop(columns = columns_to_drop)
print(df_train.shape)

print(df_test.shape)
# Remove duplicates from training and test data

df_train = df_train.drop_duplicates(subset=None, keep='first', inplace=False).copy()

df_test = df_test.drop_duplicates(subset=None, keep='first', inplace=False).copy()

print(df_train.shape)

print(df_test.shape)
# Identifying the datatypes of the variables

continous_attrib = df_train.select_dtypes(include=np.number).columns

binary_attrib = df_train[['apache_post_operative', 'arf_apache', 'cirrhosis', 'diabetes_mellitus', 'immunosuppression',

'hepatic_failure', 'leukemia', 'lymphoma', 'solid_tumor_with_metastasis', 'gcs_unable_apache',

'intubated_apache', 'ventilated_apache']].columns

continous_attrib = continous_attrib.drop(binary_attrib)

categorical_attrib =   df_train.select_dtypes(include=['object']).columns

selected_attributes = list(set(continous_attrib)) + list(set(categorical_attrib)) + list(set(binary_attrib))

print(len(selected_attributes))

df_train,y_train = df_train[selected_attributes],target

df_test= df_test[selected_attributes]

df_train.head()

df_test.head()

#Imputing features in train and test data

from sklearn.impute import SimpleImputer

# Replacing NAN values in numerical columns with mean



imputer_Num = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer_Cat_Bin = SimpleImputer(missing_values=np.nan, strategy='most_frequent')



continous_attrib = list(set(continous_attrib)) 

categorical_attrib = list(set(categorical_attrib)) 

binary_attrib = list(set(binary_attrib))



#One hot coding for train data

df_Cat_attrib = pd.DataFrame(df_train[categorical_attrib])

df_Cat_OneHotCoded = pd.get_dummies(df_Cat_attrib)

df_Cat_OneHotCoded.head()



# Fit and transform to the parameters

df_imputed_Num = pd.DataFrame(imputer_Num.fit_transform(df_train[continous_attrib]))

df_imputed_Num.columns = continous_attrib



df_imputed_Cat = pd.DataFrame(imputer_Cat_Bin.fit_transform(df_Cat_OneHotCoded[df_Cat_OneHotCoded.columns]))

df_imputed_Cat.columns = df_Cat_OneHotCoded.columns



df_imputed_binary = pd.DataFrame(imputer_Num.fit_transform(df_train[binary_attrib]))

df_imputed_binary.columns = binary_attrib



train_imputed_df = pd.concat([df_imputed_Num,df_imputed_Cat,df_imputed_binary], axis=1).dropna()





#One hot coding for test data

df_Cat_attrib = pd.DataFrame(df_test[categorical_attrib])

df_Cat_OneHotCoded = pd.get_dummies(df_Cat_attrib)

df_Cat_OneHotCoded.head()



# Fit and transform to the parameters

df_imputed_Num = pd.DataFrame(imputer_Num.fit_transform(df_test[continous_attrib]))

df_imputed_Num.columns = continous_attrib



df_imputed_Cat = pd.DataFrame(imputer_Cat_Bin.fit_transform(df_Cat_OneHotCoded[df_Cat_OneHotCoded.columns]))

df_imputed_Cat.columns = df_Cat_OneHotCoded.columns



df_imputed_binary = pd.DataFrame(imputer_Num.fit_transform(df_test[binary_attrib]))

df_imputed_binary.columns = binary_attrib



test_imputed_df = pd.concat([df_imputed_Num,df_imputed_Cat,df_imputed_binary], axis=1).dropna()



train_imputed_df.head(5)

test_imputed_df.head(5)



# The number of features in the train and test data are different.  Concat and use the get_dummies to even 

# them out such that training and test data have the same number of attributes

train_objs_num = len(train_imputed_df)

dataset = pd.concat(objs=[train_imputed_df, test_imputed_df], axis=0,sort='False')

train_X_df = dataset[:train_objs_num].copy()

test_X_df = dataset[train_objs_num:].copy()



train_X_df = train_X_df.round(decimals=2)

test_X_df = test_X_df.round(decimals=2)



train_X_df.columns.values

test_X_df.head(5)
#Distribution of train data

fig = plt.figure(figsize=(20,30))

for i in range(int(len(train_X_df.columns)-1)):

    fig.add_subplot(40,5,i+1)

    sns.distplot(train_X_df.iloc[:,i+1].dropna())

    plt.xlabel(train_X_df.columns[i])

plt.show()
threshold = 0.9

# Absolute value correlation matrix - train data

corr_matrix_train = train_X_df.corr().abs()

corr_matrix_train.head()



# test data

corr_matrix_test = test_X_df.corr().abs()

corr_matrix_test.head()
# Upper triangle of correlations - train data

upper = corr_matrix_train.where(np.triu(np.ones(corr_matrix_train.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(to_drop)

print('There are %d columns to remove.' % (len(to_drop)))

#Drop the columns with high correlations

train_X_df = train_X_df.drop(columns = to_drop)
# Upper triangle of correlations - test data

upper = corr_matrix_test.where(np.triu(np.ones(corr_matrix_test.shape), k=1).astype(np.bool))

# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(to_drop)

print('There are %d columns to remove.' % (len(to_drop)))

#Drop the columns with high correlations

test_X_df = test_X_df.drop(columns = to_drop)

train_X_df = train_X_df.drop(columns = 'd1_hematocrit_max')
print(train_X_df.shape)

print(test_X_df.shape)

from sklearn.model_selection import train_test_split

X_train, X_eval, Y_train, Y_eval = train_test_split(train_X_df, y_train, test_size=0.15, stratify=y_train)

X_train.shape, X_eval.shape, Y_train.shape, Y_eval.shape
from sklearn.model_selection import  GridSearchCV, RandomizedSearchCV

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

from xgboost import XGBClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold



from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support, roc_auc_score)



gkf = KFold(n_splits=3, shuffle=True, random_state=42).split(X=X_train, y=Y_train)

fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 500,

}

params = {

    'booster': ["gbtree"],

    'learning_rate': [0.01],

    'n_estimators': [3000],#range(1000, 2000, 3000),#range(100, 500, 100)

    'min_child_weight': [1],#1

    'gamma': [0],

    'subsample': [0.4],

    'colsample_bytree': [0.8],

    'max_depth': [4],

    "scale_pos_weight": [1],

    "reg_alpha":[1],#0.08

}

xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    silent=True,

)



gsearch = GridSearchCV(

    estimator=xgb_estimator,

    param_grid=params,

    scoring='roc_auc',

    n_jobs=-1,

    cv=gkf, verbose=3

)

xgb_model = gsearch.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
fit_params_of_xgb = {

    "early_stopping_rounds":100, 

    "eval_metric" : 'auc', 

    "eval_set" : [(X_eval,Y_eval)],

    # 'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

    'verbose': 500

}

xgb_estimator = XGBClassifier(

    objective='binary:logistic',

    silent=True,    

    booster= "gbtree",

    learning_rate= 0.01,

    n_estimators=3000,#range(1000, 2000, 3000),#range(100, 500, 100)

    min_child_weight= 1,#1

    gamma= 0,

    subsample= 0.4,

    colsample_bytree= 0.8,

    max_depth= 4,

    scale_pos_weight=1,

    reg_alpha=1,#0.08

)

xgb_estimator.fit(X=X_train, y=Y_train, **fit_params_of_xgb)

gsearch.best_params_, gsearch.best_score_
Y_pred = xgb_estimator.predict(test_X_df)

submission = pd.DataFrame({

        "encounter_id": test_df["encounter_id"],

        "hospital_death": Y_pred

    })

submission.to_csv("hospital_death.csv",index=False)
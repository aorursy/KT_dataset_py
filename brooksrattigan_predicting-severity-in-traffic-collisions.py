# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pandas-profiling
!pip install missingno
!pip install mlens

import numpy as np
import pandas as pd
import pandas_profiling
import missingno as msno

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier

# Close warnings
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/data-collusions/Data_Collisions.csv')
df.info()
df.profile_report()
# The columns below will be dropped due to reasons uncessary or duplicate information or missing values more than 95%.
drop_columns = ['X','Y','INCKEY','COLDETKEY','REPORTNO','INTKEY','LOCATION','EXCEPTRSNCODE','EXCEPTRSNDESC','SEVERITYCODE.1',
                'SEVERITYDESC','INCDATE','SDOT_COLCODE','SDOTCOLNUM','ST_COLCODE','SEGLANEKEY','CROSSWALKKEY']
df = df.drop(drop_columns, axis=1)
# In these categorical variables we only have Y as a majority and the rest is missing values.
# We can infer that if the answer was N, this column was left blank, thus we can replace missing values with N. 
df['SPEEDING'].replace(pd.np.nan,'N',inplace=True)
df['PEDROWNOTGRNT'].replace(pd.np.nan,'N',inplace=True)
df['INATTENTIONIND'].replace(pd.np.nan,'N',inplace=True)
# In this variable, we have 4 categories but normally N and 0, Y and 1 are the same categories.
df['UNDERINFL'].replace('0','N', inplace=True)
df['UNDERINFL'].replace('1','Y', inplace=True)
df['UNDERINFL'].value_counts()
# Replace 'NOT ENOUGH INFORMATION / NOT APPLICABLE' with NaN.
df['SDOT_COLDESC'].replace(to_replace='NOT ENOUGH INFORMATION / NOT APPLICABLE', value=np.nan, inplace=True)

# Decrease the number of categories by appointing categories with value counts<1000 to a new category called 'OTHER'
SDOT_COLDESC_smaller_categories = df['SDOT_COLDESC'].value_counts()[df['SDOT_COLDESC'].value_counts()<1000].index
for i in df['SDOT_COLDESC']:
    if i in SDOT_COLDESC_smaller_categories:
        df['SDOT_COLDESC'].replace(i, 'OTHER', inplace=True)
df['SDOT_COLDESC'].value_counts()
# Decrease the number of categories by appointing categories with value counts<2000 to a new category called 'OTHER'
df['ST_COLDESC'].value_counts()[df['ST_COLDESC'].value_counts()<2000]
ST_COLDESC_smaller_categories = df['ST_COLDESC'].value_counts()[df['ST_COLDESC'].value_counts()<2000].index
for i in df['ST_COLDESC']:
    if i in ST_COLDESC_smaller_categories:
        df['ST_COLDESC'].replace(i, 'OTHER', inplace=True)
df['ST_COLDESC'].value_counts()
# Convert incident date from object to date, extract day and time from INCDTTM column and create new columns. 
# Some of the dates have no hour and the time is converted as 0, thus they are converted into nan values. 
df['INCDTTM'] = pd.to_datetime(df['INCDTTM'])
df['DAY'] = df['INCDTTM'].dt.day_name()
df['TIME'] = df['INCDTTM'].dt.hour
df['TIME'].replace(0, np.nan, inplace=True)
df['MONTH'] = df['INCDTTM'].dt.month
df['MONTH'].replace({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',
                        8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}, inplace=True)
df.info()
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
df_var = df.drop(['OBJECTID','INCDTTM'], axis=1)
for col in df_var.columns:
    temp = df_var[col].value_counts()
    df1 = pd.DataFrame({'labels': temp.index,
                       'values': temp.values
                        })
    df1.iplot(kind='pie',labels='labels',values='values', title=col, hole = 0.5)
for i, col in enumerate(df_var):
    plt.figure(i)
    plt.xticks(rotation= 90)
    sns.countplot(x=col, hue='SEVERITYCODE', data=df_var)
plt.figure(figsize = (10, 8))

sns.kdeplot(df.loc[df['SEVERITYCODE'] == 1, 'TIME'], label = 'target == 1')

sns.kdeplot(df.loc[df['SEVERITYCODE'] == 2, 'TIME'], label = 'target == 2')

# Labeling of plot
plt.xlabel('TIME'); plt.ylabel('Density'); plt.title('DISTRIBUTION OF TIME BY SEVERITY');
plt.figure(figsize = (10, 8))

sns.kdeplot(df.loc[df['SEVERITYCODE'] == 1, 'PERSONCOUNT'], label = 'target == 1')

sns.kdeplot(df.loc[df['SEVERITYCODE'] == 2, 'PERSONCOUNT'], label = 'target == 2')

# Labeling of plot
plt.xlabel('PERSONCOUNT'); plt.ylabel('Density'); plt.title('DISTRIBUTION OF PERSONCOUNT BY SEVERITY');
fig, ax = plt.subplots(figsize=(12,8)) 
sns.heatmap(df.iloc[:,0:len(df_var)].corr(), annot = True, fmt = ".2f", linewidths=0.5, ax=ax) 
plt.show()
# Percentages of NaN values after data manipulation. 
# it looks that the percentages of NaN values are reasonable.
total_nan = df.isnull().sum().sort_values(ascending = False)
percent_nan = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
missing_df  = pd.concat([total_nan, percent_nan], axis=1, keys=['Total_nan', 'Percent_nan'])
missing_df.head(10)
msno.matrix(df);
msno.bar(df);
# Nullity correlation
msno.heatmap(df);
# binary label encoding
lbe = LabelEncoder()
df['STATUS'] = lbe.fit_transform(df['STATUS'])
df['INATTENTIONIND'] = lbe.fit_transform(df['INATTENTIONIND'])
df['PEDROWNOTGRNT'] = lbe.fit_transform(df['PEDROWNOTGRNT'])
df['SPEEDING'] = lbe.fit_transform(df['SPEEDING'])
df['HITPARKEDCAR'] = lbe.fit_transform(df['HITPARKEDCAR'])

# use replace method since we have nan values in this column and Labelencoder() gives error.  
df['UNDERINFL'].replace({'Y':1, 'N':0}, inplace=True)
df_cat_columns = df.select_dtypes(include=object).columns
df_cat_columns
# one hot encoding
df = pd.get_dummies(df, columns=df_cat_columns, drop_first=True)
# some of the category names have spaces, they need to corrected to encounter errors in the ML models
import re
df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '_', x))
df.drop(['OBJECTID', 'INCDTTM',], axis=1, inplace=True)
# Making a copy of df with null values
df_with_null = df.copy()
df.columns = df_with_null.columns
# KNN imputation of nan values
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns = df_with_null.columns)
# Features and Target Variable

X = df.drop(['SEVERITYCODE'], axis=1)
y = df['SEVERITYCODE']
# Split the data into training/testing sets

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, shuffle=y, stratify=y, random_state = 42)
# Set cross validation, we use stratified since the target variable is imbalanced

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
# Data with Null Values

X_with_null = df_with_null.drop(['SEVERITYCODE'], axis=1)
y_with_null = df_with_null['SEVERITYCODE']

X_train_with_null, X_test_with_null, y_train_with_null, y_test_with_null= train_test_split(
    X_with_null, y_with_null, test_size=0.2, shuffle=y, stratify=y, random_state = 42)
# Create the model
log_reg = LogisticRegression(random_state=42)

# Fit the model
log_reg.fit(X_train, y_train)

# Predict the model
y_pred = log_reg.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
#  Model tuning

log_reg_params = {"C":np.logspace(-1, 1, 10),
                  "penalty": ["l1","l2"], 
                  "solver":['lbfgs', 'liblinear', 'sag', 'saga']}
log_reg = LogisticRegression(random_state=42)
log_reg_cv_model = RandomizedSearchCV(log_reg, log_reg_params, cv=sss)
log_reg_cv_model.fit(X_train, y_train)
print("Best score:" + str(log_reg_cv_model.best_score_))
print("Best parameters: " + str(log_reg_cv_model.best_params_))
# Create the model
rf = RandomForestClassifier(random_state=42)

# Fit the model
rf_model = rf.fit(X_train, y_train)

# Predict the model
y_pred = rf_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

rf_params = {"max_depth": [2,5,8,10],
            "max_features": [2,3,5,7],
            "n_estimators": [10,100,200,500,1000],
            "min_samples_split": [2,5,10]}
rf_model = RandomForestClassifier(random_state=42)
rf_cv_model = RandomizedSearchCV(rf_model, 
                           rf_params, 
                           cv = sss, 
                           n_jobs = -1, 
                           verbose = 2) 
rf_cv_model.fit(X_train, y_train)
print("Best score:" + str(rf_cv_model.best_score_))
print("Best Parameters: " + str(rf_cv_model.best_params_))
# Create the model
lgbm = LGBMClassifier(random_state=42)

# Fit the model
lgbm_model = lgbm.fit(X_train, y_train)

# Predict the model
y_pred = lgbm_model.predict(X_test)

# Accuracy Score
accuracy_score(y_test, y_pred)
# Model Tuning

lgbm_params = {
        'n_estimators': [100, 500, 1000, 1500, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.01, 0.02, 0.05],
        "min_child_samples": [5, 10, 20]}
lgbm = LGBMClassifier(random_state=42)
lgbm_cv_model = RandomizedSearchCV(lgbm, lgbm_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
lgbm_cv_model.fit(X_train, y_train)
print("Best score:" + str(lgbm_cv_model.best_score_))
print("Best parameters: " + str(lgbm_cv_model.best_params_))
# Create the model
lgbm = LGBMClassifier(random_state=42)

# Fit the model
lgbm_model = lgbm.fit(X_train_with_null, y_train_with_null)

# Predict the model
y_pred_with_null = lgbm_model.predict(X_test_with_null)

# Accuracy Score
accuracy_score(y_test_with_null, y_pred_with_null)
# Model Tuning

lgbm_params = {
        'n_estimators': [100, 500, 1000, 1500, 2000],
        'subsample': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.1, 0.01, 0.02, 0.05],
        "min_child_samples": [5, 10, 20]}
lgbm = LGBMClassifier(random_state=42)
lgbm_cv_model_with_null = RandomizedSearchCV(lgbm, lgbm_params, 
                             cv = sss, 
                             n_jobs = -1, 
                             verbose = 2)
lgbm_cv_model_with_null.fit(X_train_with_null, y_train_with_null)
print("Best score:" + str(lgbm_cv_model_with_null.best_score_))
print("Best parameters: " + str(lgbm_cv_model_with_null.best_params_))
# Tuned Logistic Regression Model

param = log_reg_cv_model.best_params_
log_reg = LogisticRegression(**param, random_state=42)
log_reg_tuned = log_reg.fit(X_train, y_train)
y_pred = log_reg_tuned.predict(X_test)
log_reg_final = accuracy_score(y_test, y_pred)
log_reg_final

# Tuned Random Forest Model 

param = rf_cv_model.best_params_
rf_tuned = RandomForestClassifier(**param, random_state=42)
rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
rf_final = accuracy_score(y_test, y_pred)
rf_final
# Tuned LGBM Model

param = lgbm_cv_model.best_params_
lgbm = LGBMClassifier(**param, random_state=42)
lgbm_tuned = lgbm.fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
lgbm_final = accuracy_score(y_test, y_pred)
lgbm_final
# feature importances with LGBM

feature_importances = pd.DataFrame({'feature': list(X_train.columns), 
                                    'importance': lgbm_tuned.feature_importances_}).sort_values('importance', ascending = False)

feature_importances.head(10)
# Tuned LGBM Model - Handling Missing Values Internally

param = lgbm_cv_model_with_null.best_params_
lgbm_with_null = LGBMClassifier(**param, random_state=42)
lgbm_tuned = lgbm_with_null.fit(X_train_with_null, y_train_with_null)
y_pred_with_null = lgbm_tuned.predict(X_test_with_null)
lgbm_final_with_null = accuracy_score(y_test_with_null, y_pred_with_null)
lgbm_final_with_null
accuracy_scores = {
'log_reg_final': log_reg_final,
'rf_final': rf_final,
'lgbm_final': lgbm_final,
'lgbm_final_with_null': lgbm_final_with_null,
}

accuracy_scores = pd.Series(accuracy_scores).to_frame('Accuracy_Score')
accuracy_scores = accuracy_scores.sort_values(by='Accuracy_Score', ascending=False)
accuracy_scores['rank'] = (accuracy_scores.reset_index().index +1)
accuracy_scores
model = [('Logistic Regression', log_reg),
         ('LGBM', lgbm),
         ('Random Forest', rf)
          ]
voting_reg = VotingClassifier(model, voting='hard')
voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)
print(f"Voting Classifier's accuracy: {accuracy_score(y_pred, y_test):.4f}")
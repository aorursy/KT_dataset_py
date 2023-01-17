# This Python 3 environment 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, make_scorer, recall_score, precision_score, f1_score, roc_auc_score



from sklearn.model_selection import GridSearchCV
!pip install statsmodels==0.10.0rc2 --pre  # Statsmodel has sme problem with factorial in latest lib
# Install the Library (Refer: https://pypi.org/project/kesh-utils/ )

!pip install kesh-utils
# Ignore the warnings if any

import warnings  

warnings.filterwarnings('ignore')
# Load the dataset 

adult_income_df = pd.read_csv('../input/adult.csv')
adult_income_df.head()
# Quick known cleanup for this dataset

adult_income_df['workclass']=adult_income_df['workclass'].replace('?','Unknown') # Treat ? workclass as unknown

adult_income_df = adult_income_df[adult_income_df['occupation'] != '?'] # Remove rows with occupation =?

adult_income_df['native.country']=adult_income_df['native.country'].replace('?', adult_income_df['native.country'].mode()[0]) # Replace ? with mode

adult_income_df['fnlwgt']=np.log(adult_income_df['fnlwgt']) # Convert to antural log

adult_income_df.loc[adult_income_df['native.country']!='United-States','native.country'] = 'non_usa' # Two many category level, convert just US and Non-US
# We will use Label encoder for all categorical variables

from sklearn import preprocessing



# encode categorical variables using Label Encoder

# select all categorical variables

df_categorical = adult_income_df.select_dtypes(include=['object'])

df_categorical.head()



# apply Label encoder to df_categorical

le = preprocessing.LabelEncoder()

df_categorical = df_categorical.apply(le.fit_transform)

df_categorical.head()



# concat df_categorical with original df

adult_income_df = adult_income_df.drop(df_categorical.columns, axis=1)

adult_income_df = pd.concat([adult_income_df, df_categorical], axis=1)
# Scale the numerical features using StandardScalar

from sklearn.preprocessing import StandardScaler

numerical_column_names = ['age','fnlwgt','education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

scaler = StandardScaler()



adult_income_df[numerical_column_names] = scaler.fit_transform(

    adult_income_df[numerical_column_names])
# Final cleaned dataset 

adult_income_df.head()
# Prepare the data for model building and evaluation

X = adult_income_df.drop('income', axis=1)

y = adult_income_df['income'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=43)
# Load the custom library

from KUtils.classifier import generic_classifier_utils as gcu
from sklearn.tree import DecisionTreeClassifier



scores = gcu.single_hyperparameter_multiple_scoring_tuning(

    X_train, y_train,

    cv_folds=5, 

    hyper_parameter_name='max_depth',

    hyper_parameter_range = range(3, 21, 3),

    model_scoring = {'F1': make_scorer(f1_score),

                     'AUC': make_scorer(roc_auc_score),

                     'Accuracy': make_scorer(accuracy_score)        #  'Accuracy': make_scorer(accuracy_score),

                    },

    refit='AUC',

    classifier_algo=DecisionTreeClassifier())
from sklearn.ensemble import RandomForestClassifier



scores = gcu.single_hyperparameter_multiple_scoring_tuning(

    X_train, y_train,

    cv_folds=10, 

    hyper_parameter_name='n_estimators',

    hyper_parameter_range =range(5, 200, 25),   

    model_scoring = {'F1': make_scorer(f1_score),

                     'AUC': make_scorer(roc_auc_score),

                     'Accuracy': make_scorer(accuracy_score)        #  'Accuracy': make_scorer(accuracy_score),

                    },

    refit='AUC',

    classifier_algo=RandomForestClassifier(max_depth=4))
from xgboost.sklearn import XGBClassifier



scores = gcu.single_hyperparameter_multiple_scoring_tuning(

    X_train, y_train,

    cv_folds=10, 

    hyper_parameter_name='learning_rate',

    hyper_parameter_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9],

    model_scoring = {'F1': make_scorer(f1_score),

                     'AUC': make_scorer(roc_auc_score),

                     'Accuracy': make_scorer(accuracy_score)        #  'Accuracy': make_scorer(accuracy_score),

                    },

    refit='AUC',

    classifier_algo=XGBClassifier(objective= 'binary:logistic'))
import lightgbm as lgb



scores = gcu.single_hyperparameter_multiple_scoring_tuning(

    X_train, y_train,

    cv_folds=10,

    hyper_parameter_name='num_leaves',

    hyper_parameter_range = [2, 5, 10, 50, 100, 200],

    model_scoring = {'F1': make_scorer(f1_score),

                     'AUC': make_scorer(roc_auc_score),

                     'Accuracy': make_scorer(accuracy_score)        #  'Accuracy': make_scorer(accuracy_score),

                    },

    refit='Accuracy',

    classifier_algo=lgb.LGBMClassifier(n_jobs=-1))
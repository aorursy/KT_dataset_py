# Starter code to load data

import pandas as pd

# Training dataset

data=pd.read_csv('../input/bgse-svm-death/mimic_train.csv')

data.head()
# Test dataset (to produce predictions)

data_test=pd.read_csv('../input/bgse-svm-death/mimic_test_death.csv')

data_test.sort_values('icustay_id').head()
# Sample output prediction file

pred_sample=pd.read_csv('../input/bgse-svm-death/mimic_kaggle_death_sample_submission.csv')

pred_sample.sort_values('icustay_id').head()
#your code here
#!pip install ipython-autotime
import pandas as pd

import numpy as np

import time



from sklearn.svm import SVC, LinearSVC



np.random.seed(3123) # impose random seed for reproducibility



import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import make_column_transformer, ColumnTransformer

from sklearn.pipeline import Pipeline, make_pipeline



# kernel approximators

from sklearn.kernel_approximation import Nystroem, RBFSampler



from imblearn.under_sampling import  RandomUnderSampler

from imblearn.over_sampling import  RandomOverSampler



import gc
%%time



exp_flag = data['HOSPITAL_EXPIRE_FLAG']

numbers = ['HeartRate_Min','HeartRate_Max','HeartRate_Mean','SysBP_Min',

       'SysBP_Max', 'SysBP_Mean', 'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',

       'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean', 'RespRate_Min',

       'RespRate_Max', 'RespRate_Mean', 'TempC_Min', 'TempC_Max', 'TempC_Mean',

       'SpO2_Min', 'SpO2_Max', 'SpO2_Mean', 'Glucose_Min', 'Glucose_Max',

       'Glucose_Mean']

categ = ['ADMISSION_TYPE', 'FIRST_CAREUNIT']



X=data.loc[:,numbers+categ]

X_test=data_test.loc[:,numbers+categ]



# Split preprocessing depending on type

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])



categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numbers),

        ('cat', categorical_transformer, categ)],

        remainder='drop')



preprocessor.fit(X)

X=preprocessor.transform(X)

X_test=preprocessor.transform(X_test)



X=pd.DataFrame(X, 

              columns= numbers+

               list(preprocessor.transformers_[1][1]['onehot'].get_feature_names(categ)))

X_test=pd.DataFrame(X_test, 

              columns= numbers+

               list(preprocessor.transformers_[1][1]['onehot'].get_feature_names(categ)))



X_colnames=X.columns

X.head()
%%time

# Undersample

sampler = RandomUnderSampler(random_state=0)

X, exp_flag = sampler.fit_resample(X, exp_flag)
%%time

# Try linear kernel using LinearSVC

MySvc = LinearSVC( max_iter=10000)

grid_values = {'C':[0.1, 1, 10]}

grid_svc_acc1 = GridSearchCV(MySvc, 

                    param_grid = grid_values, scoring='roc_auc',

                    cv=StratifiedKFold(n_splits=5, shuffle=True),

                    n_jobs=5)

grid_svc_acc1.fit(X, exp_flag)

grid_svc_acc1.best_score_

%%time

# Try linear kernel

gc.collect()



MySvc = SVC(kernel='linear')

grid_values = {'C':[0.1, 1, 10]}

grid_svc_acc2 = GridSearchCV(MySvc, 

                    param_grid = grid_values, scoring='roc_auc',

                    cv=StratifiedKFold(n_splits=5, shuffle=True),

                    n_jobs=5)

grid_svc_acc2.fit(X, exp_flag)

grid_svc_acc2.best_score_
%%time

# Try  rbf

gc.collect()



MySvc = SVC(kernel='rbf')

grid_values = {'C':[0.1, 1, 10]}

grid_svc_acc3 = GridSearchCV(MySvc, 

                    param_grid = grid_values, scoring='roc_auc',

                    cv=StratifiedKFold(n_splits=5, shuffle=True),

                    n_jobs=5)

grid_svc_acc3.fit(X, exp_flag)

grid_svc_acc3.best_score_
%%time

gc.collect()



# Try kernel approximation with Nystroem

GAMMA=0.2

N_Comp=200

feature_map_nystroem = Nystroem(gamma=GAMMA,

                                 random_state=1,

                                 n_components=N_Comp)

feature_map_nystroem.fit(X)

X_transformed = feature_map_nystroem.transform(X)

MySvc = LinearSVC( max_iter=10000)

grid_values = {'C':[0.1, 1, 10]}

grid_svc_acc4 = GridSearchCV(MySvc, 

                    param_grid = grid_values, scoring='roc_auc',

                    cv=StratifiedKFold(n_splits=5, shuffle=True),

                    n_jobs=5)

grid_svc_acc4.fit(X, exp_flag)

grid_svc_acc4.best_score_
%%time

# Try kernel approximation with RBFSampler

gc.collect()



feature_map_sampler = RBFSampler(gamma=GAMMA,

                                 random_state=1,

                                 n_components=N_Comp)

feature_map_sampler.fit(X)

X_transformed = feature_map_sampler.transform(X)

MySvc = LinearSVC( max_iter=10000)

grid_values = {'C':[0.1, 1, 10]}

grid_svc_acc5 = GridSearchCV(MySvc, 

                    param_grid = grid_values, scoring='roc_auc',

                    cv=StratifiedKFold(n_splits=5, shuffle=True),

                    n_jobs=5)

grid_svc_acc5.fit(X, exp_flag)

grid_svc_acc5.best_score_
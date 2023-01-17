import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from tqdm import tqdm

import gc



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Train.csv")

health_camp = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Health_Camp_Detail.csv")

patient_profile = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Patient_Profile.csv")

first_hc_attended = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/First_Health_Camp_Attended.csv")

second_hc_attended = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Second_Health_Camp_Attended.csv")

third_hc_attended = pd.read_csv("../input/janatahackhealthcareanalytics/Train_2/Train/Third_Health_Camp_Attended.csv")



test = pd.read_csv("../input/janatahackhealthcareanalytics/test_l0Auv8Q.csv")

submission = pd.read_csv("../input/janatahackhealthcareanalytics/sample_submmission.csv")
def data_merging(df):

    df_pp = pd.merge(df,

                  patient_profile,

                  on = 'Patient_ID',

                  how = 'left'

                 )

    df_pp_hc = pd.merge(df_pp,

                       health_camp,

                       on = 'Health_Camp_ID',

                       how = 'left')

    

    return df_pp_hc
def generate_train_labels(df_pp_hc):

    df_pp_hc_1 = pd.merge(df_pp_hc,

                         first_hc_attended,

                         on = ['Patient_ID', 'Health_Camp_ID'],

                         how = 'left')

    df_pp_hc_12 = pd.merge(df_pp_hc_1,

                         second_hc_attended,

                         on = ['Patient_ID', 'Health_Camp_ID'],

                         how = 'left')

    df_pp_hc_123 = pd.merge(df_pp_hc_12,

                         third_hc_attended,

                         on = ['Patient_ID', 'Health_Camp_ID'],

                         how = 'left')

    

    df_pp_hc_123['Outcome'] = df_pp_hc_123.apply(lambda x: 1 if (x['Health_Score'] > 0 

                                                                 or x['Health Score'] > 0 

                                                                 or x['Number_of_stall_visited'] > 0) 

                                                             else 0,

                                                    axis=1)

    

    df_pp_hc_123 = df_pp_hc_123.drop(['Patient_ID', 'Health_Camp_ID', 'Donation', 'Health_Score', 'Health Score', 'Number_of_stall_visited', 'Last_Stall_Visited_Number'], axis=1)

    

    return df_pp_hc_123
def data_preprocessing(df):

    df['Registration_Date'].fillna(df['Registration_Date'].mode()[0], inplace=True)

    df['Income'] = df['Income'].replace('None', np.nan)

    df['Income'].fillna(df['Income'].mode()[0], inplace=True)

    df['Education_Score'] = df['Education_Score'].replace('None', np.nan)

    df['Education_Score'].fillna(df['Education_Score'].mode()[0], inplace=True)

    df['Age'] = df['Age'].replace('None', np.nan)

    df['Age'].fillna(df['Age'].mode()[0], inplace=True)

    df['City_Type'] = df['City_Type'].fillna('Unknown')

    df['Employer_Category'] = df['Employer_Category'].fillna('Unknown')

    df['Income'] = df['Income'].astype("int16")

    df['Education_Score'] = df['Education_Score'].astype("float64")

    df['Age'] = df['Age'].astype("int16")

    

    catcols = ['Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared', 'Category1', 'Category2', 'Category3', 'City_Type', 'Employer_Category']



    for col in catcols:

        df[col] = df[col].astype('category')

        df[col] = df[col].cat.codes.astype("int16")

        

    return df
def date_features(df):

    df['Registration_Date'] = pd.to_datetime(df['Registration_Date'], dayfirst=True)

    df['First_Interaction'] = pd.to_datetime(df['First_Interaction'], dayfirst=True)

    df['Camp_Start_Date'] = pd.to_datetime(df['Camp_Start_Date'], dayfirst=True)

    df['Camp_End_Date'] = pd.to_datetime(df['Camp_End_Date'], dayfirst=True)

    df['Camp_Length'] = (df['Camp_End_Date'] - df['Camp_Start_Date']).dt.days

    df['CampStart_Reg_Gap'] = (df['Registration_Date'] - df['Camp_Start_Date']).dt.days

    df['Reg_CampEnd_Gap'] = (df['Camp_End_Date'] - df['Registration_Date']).dt.days

    df['FI_Reg_Gap'] = (df['Registration_Date'] - df['First_Interaction']).dt.days

    df['CampStart_FI_Gap'] = (df['Camp_Start_Date'] - df['First_Interaction']).dt.days

    df['CampEnd_FI_Gap'] = (df['Camp_End_Date'] - df['First_Interaction']).dt.days



    reg_date_features = {



                        "reg_weekday": "weekday",

                        "reg_weekofyear": "weekofyear",

                        "reg_month": "month",

                        "reg_quarter": "quarter",

                        "reg_year": "year",

                        "reg_mday": "day",

                        "reg_ime": "is_month_end",

                        "reg_ims": "is_month_start",

                    }

    

    for date_feat_name, date_feat_func in reg_date_features.items():

        df[date_feat_name] = getattr(df["Registration_Date"].dt, date_feat_func).astype("int16")

        

    fi_date_features = {



                        "fi_weekday": "weekday",

                        "fi_weekofyear": "weekofyear",

                        "fi_month": "month",

                        "fi_quarter": "quarter",

                        "fi_year": "year",

                        "fi_mday": "day",

                        "fi_ime": "is_month_end",

                        "fi_ims": "is_month_start",

                    }

    

    for date_feat_name, date_feat_func in fi_date_features.items():

        df[date_feat_name] = getattr(df["First_Interaction"].dt, date_feat_func).astype("int16")

                

    csd_date_features = {



                        "csd_weekday": "weekday",

                        "csd_weekofyear": "weekofyear",

                        "csd_month": "month",

                        "csd_quarter": "quarter",

                        "csd_year": "year",

                        "csd_mday": "day",

                        "csd_ime": "is_month_end",

                        "csd_ims": "is_month_start",

                    }

    

    for date_feat_name, date_feat_func in csd_date_features.items():

        df[date_feat_name] = getattr(df["Camp_Start_Date"].dt, date_feat_func).astype("int16")

    

    ced_date_features = {



                        "ced_weekday": "weekday",

                        "ced_weekofyear": "weekofyear",

                        "ced_month": "month",

                        "ced_quarter": "quarter",

                        "ced_year": "year",

                        "ced_mday": "day",

                        "ced_ime": "is_month_end",

                        "ced_ims": "is_month_start",

                    }

    

    for date_feat_name, date_feat_func in ced_date_features.items():

        df[date_feat_name] = getattr(df["Camp_End_Date"].dt, date_feat_func).astype("int16") 

        

    df.drop(['Registration_Date', 'First_Interaction', 'Camp_Start_Date', 'Camp_End_Date'], axis=1, inplace=True)

        

    return df
train.info()
patient_profile.info()
health_camp.info()
train_pp_hc = data_merging(train)
train_pp_hc.info()
# Removing last column as it's null column in data

first_hc_attended = first_hc_attended.iloc[:, :-1]

first_hc_attended.info()
second_hc_attended.info()
third_hc_attended.info()
train_pp_hc_123 = generate_train_labels(train_pp_hc)

train_pp_hc_123.info()
print(train_pp_hc_123.Outcome.sum())

print(train_pp_hc_123.Outcome.value_counts())
train_pp_hc_123.head()
train_pp_hc_123.Income.value_counts(normalize=True)
train_pp_hc_123.Education_Score.value_counts(normalize=True)
train_pp_hc_123.Age.value_counts(normalize=True)
train_pp_hc_123.columns
train_pp_hc_123[['Registration_Date', 'First_Interaction', 'Camp_Start_Date', 'Camp_End_Date']].head(10)
train_df = data_preprocessing(train_pp_hc_123)

train_df = date_features(train_df)
train_df.info()
train_df.Outcome.value_counts()
train_df.columns
categorical_features = ['Online_Follower', 'LinkedIn_Shared', 'Twitter_Shared', 'Facebook_Shared']

# categorical_features = ['Category1']

less_importance_features = ['csd_month', 'ced_ims', 'fi_ims', 'reg_ime', 'fi_ime', 'ced_month', 'Var4', 'Category3', 'Facebook_Shared', 'Twitter_Shared', 

'ced_year', 'csd_year', 'reg_year', 'Online_Follower', 'reg_month', 'LinkedIn_Shared', 'Var2', 'Category2', 'fi_year', 'Var5', 'ced_quarter', 'reg_quarter', 

'fi_month', 'csd_mday']

# less_importance_features = []

label_col = ['Outcome']

useless_cols = less_importance_features + label_col

train_cols = train_df.columns[~train_df.columns.isin(useless_cols)]

X_train_df = train_df[train_cols]

y_train_df = train_df['Outcome']
X_train_df.shape
X_train, X_test, y_train, y_test = train_test_split(X_train_df, y_train_df, stratify = y_train_df, test_size=0.20, random_state=42)
def learning_rate_010_decay_power_099(current_iter):

    base_learning_rate = 0.1

    lr = base_learning_rate  * np.power(.99, current_iter)

    return lr if lr > 1e-3 else 1e-3



def learning_rate_010_decay_power_0995(current_iter):

    base_learning_rate = 0.1

    lr = base_learning_rate  * np.power(.995, current_iter)

    return lr if lr > 1e-3 else 1e-3



def learning_rate_005_decay_power_099(current_iter):

    base_learning_rate = 0.05

    lr = base_learning_rate  * np.power(.99, current_iter)

    return lr if lr > 1e-3 else 1e-3



def learning_rate_003_decay_power_099(current_iter):

    base_learning_rate = 0.03

    lr = base_learning_rate  * np.power(.99, current_iter)

    return lr if lr > 1e-3 else 1e-3
fit_params={"early_stopping_rounds":30, 

            "eval_metric" : 'auc', 

            "eval_set" : [(X_test,y_test)],

            'eval_names': ['valid'],

            #'callbacks': [lgb.reset_parameter(learning_rate=learning_rate_010_decay_power_099)],

            'verbose': 100,

            'categorical_feature': 'auto'}
from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

param_test ={'num_leaves': sp_randint(6, 50), 

             'min_child_samples': sp_randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': sp_uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}
n_HP_points_to_test = 100



import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV



#n_estimators is set to a "large value". The actual number of trees build will depend on early stopping and 5000 define only the absolute maximum

clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=5000)

gs = RandomizedSearchCV(

    estimator=clf, param_distributions=param_test, 

    n_iter=n_HP_points_to_test,

    scoring='roc_auc',

    cv=3,

    refit=True,

    random_state=42,

    verbose=True)
%%time

gs.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs.best_score_, gs.best_params_))
# gs.booster_.save_model(save_to)
# opt_params = {'colsample_bytree': 0.8665631328558623, 'min_child_samples': 122, 'min_child_weight': 0.1, 'num_leaves': 48, 'reg_alpha': 2, 'reg_lambda': 50, 'subsample': 0.7252600946741159}

# opt_params = {'colsample_bytree': 0.7121008659272392, 'min_child_samples': 207, 'min_child_weight': 0.001, 'num_leaves': 40, 'reg_alpha': 0.1, 'reg_lambda': 20, 'subsample': 0.20019001886070767}

opt_params = gs.best_params_
clf_sw = lgb.LGBMClassifier(**clf.get_params())

#set optimal parameters

clf_sw.set_params(**opt_params)
gs_sample_weight = GridSearchCV(estimator=clf_sw, 

                                param_grid={'scale_pos_weight':[1,2,6,12]},

                                scoring='roc_auc',

                                cv=5,

                                refit=True,

                                verbose=True,

                                return_train_score=True)
%%time

gs_sample_weight.fit(X_train, y_train, **fit_params)

print('Best score reached: {} with params: {} '.format(gs_sample_weight.best_score_, gs_sample_weight.best_params_))
# gs_sample_weight.booster_.save_model(save_to)
print("Valid+-Std     Train  :   Parameters")

for i in np.argsort(gs_sample_weight.cv_results_['mean_test_score'])[-5:]:

    print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(gs_sample_weight.cv_results_['params'][i], 

                                    gs_sample_weight.cv_results_['mean_test_score'][i], 

                                    gs_sample_weight.cv_results_['mean_train_score'][i],

                                    gs_sample_weight.cv_results_['std_test_score'][i]))
%%time

#Configure from the HP optimisation

#clf_final = lgb.LGBMClassifier(**gs.best_estimator_.get_params())



#Configure locally from hardcoded values

clf_final = lgb.LGBMClassifier(**clf.get_params())

#set optimal parameters

clf_final.set_params(**opt_params)



#Train the final model with learning rate decay

clf_final.fit(X_train, y_train, **fit_params, callbacks=[lgb.reset_parameter(learning_rate=learning_rate_005_decay_power_099)])
# clf_final.booster_.save_model(save_to)
feat_imp = pd.Series(clf_final.feature_importances_, index=train_cols)

feat_imp.nlargest(20).plot(kind='barh', figsize=(12,16)).invert_yaxis()
test_pp_hc = data_merging(test)

test_pp_hc[['Patient_ID', 'Health_Camp_ID']].groupby(['Patient_ID', 'Health_Camp_ID']).agg('count')

test_pp_hc = data_preprocessing(test_pp_hc)

test_pp_hc = date_features(test_pp_hc)

test_pp_hc.info()
test_df = test_pp_hc[train_cols]

test_df.info()
probabilities = clf_final.predict_proba(test_df, axis=1)

submission = pd.DataFrame({

    'Patient_ID': submission['Patient_ID'],

    'Health_Camp_ID': submission['Health_Camp_ID'],

    'Outcome':     [ row[1] for row in probabilities]

})

submission.to_csv("lgb_gridsearch_cv3_hpitr100_lrshr_005_099_corrected3rdHCNoStall_lessfeatures.csv", index=False)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', None)  



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import h2o

import pandas as pd

from h2o.automl import H2OAutoML, get_leaderboard

from sklearn.model_selection import StratifiedShuffleSplit
df = pd.read_csv('/kaggle/input/auto-insurance-claims-data/insurance_claims.csv')



print(df.shape)

df.head()
df.isnull().sum()
# drop column _c39 because it's null



df = df.drop(['_c39'], axis = 1)
df.dtypes
# Set policy_bind_date type as date

df['policy_bind_date'] = pd.to_datetime(df['policy_bind_date'])

df['incident_date'] = pd.to_datetime(df['incident_date'])
# How long before accident

df['incident_date2policy_bind_date'] = (df['incident_date'] - df['policy_bind_date']).dt.days
# Dummies columns

                       

policy_state_dummies = pd.get_dummies(df['policy_state'], drop_first=True, prefix='policy_state')

incident_type_dummies = pd.get_dummies(df['incident_type'], drop_first=True, prefix='incident_type')

insured_zip_dummies = pd.get_dummies(df['insured_zip'], drop_first=True, prefix='insured_zip')

insured_education_level_dummies = pd.get_dummies(df['insured_education_level'], drop_first=True, prefix='insured_education_level')

insured_sex_dummies = pd.get_dummies(df['insured_sex'], drop_first=True, prefix='insured_sex')

insured_occupation_dummies = pd.get_dummies(df['insured_occupation'], drop_first=True, prefix='insured_occupation')

insured_hobbies_dummies = pd.get_dummies(df['insured_hobbies'], drop_first=True, prefix='insured_hobbies')

insured_relationship_dummies = pd.get_dummies(df['insured_relationship'], drop_first=True, prefix='insured_relationship')

incident_severity_dummies = pd.get_dummies(df['incident_severity'], drop_first=True, prefix='incident_severity')

authorities_contacted_dummies = pd.get_dummies(df['authorities_contacted'], drop_first=True, prefix='authorities_contacted')

incident_city_dummies = pd.get_dummies(df['incident_city'], drop_first=True, prefix='incident_city')

auto_make_dummies = pd.get_dummies(df['auto_make'], drop_first=True, prefix='auto_make')





collision_type_dummies = pd.get_dummies(df['collision_type'].replace('?','missing'), drop_first=True, prefix='collision_type')

property_damage_dummies = pd.get_dummies(df['property_damage'].replace('?','missing'), drop_first=True, prefix='property_damage')

police_report_available_dummies = pd.get_dummies(df['police_report_available'].replace('?','missing'), drop_first=True, prefix='police_report_available')



# set the target

target_cols = pd.get_dummies(df['fraud_reported'], drop_first=True).astype(str)

target_cols.columns = ['target']
# Drop unuse columns



drop_cols = ['policy_number','policy_bind_date','policy_csl','incident_date','auto_make','incident_location']

dummies_cols = ['insured_zip','insured_education_level','insured_sex','insured_occupation','insured_hobbies','insured_relationship','incident_type',

 'incident_severity','authorities_contacted','incident_city','auto_model','collision_type','property_damage','fraud_reported',

               'police_report_available','policy_state','incident_state']



df = df.drop(drop_cols+dummies_cols, axis=1)
df.dtypes
# Add dummies to data frame

dummies_df = [policy_state_dummies,insured_zip_dummies,insured_education_level_dummies,insured_sex_dummies,insured_occupation_dummies,

             insured_hobbies_dummies,insured_relationship_dummies,incident_severity_dummies,authorities_contacted_dummies,

             auto_make_dummies,collision_type_dummies,property_damage_dummies,police_report_available_dummies, target_cols]



df = pd.concat([df] + dummies_df, axis=1)
df.shape
df.head()
# Init H20

h2o.init()
# train text split

seed = 56



hf = h2o.H2OFrame(df)

hf['target'] =hf['target'].asfactor()



train,test,valid = hf.split_frame(ratios=[0.7, 0.15])
hf.shape


target_label = 'target'

features_list = [x for x in hf.columns if x != target_label]
# Train model

aml = H2OAutoML(max_runtime_secs=60 * 30, include_algos=["XGBoost"], seed=seed)



aml.train(x=features_list, y=target_label, training_frame=train)
h2o_result = get_leaderboard(aml, extra_columns='ALL').as_data_frame().loc[0]



model_name = h2o_result['model_id']

best_model = h2o.get_model(model_name)



preds = best_model.predict(test[:-1])

global_predict_res = preds.as_data_frame()

run_time = h2o_result['training_time_ms']
preds
h2o_result.auc
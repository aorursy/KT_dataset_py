# # This Python 3 environment comes with many helpful analytics libraries installed

# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# # For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# # Input data files are available in the read-only "../input/" directory

# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# # You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# !pip install xarray==0.16.0

# !pip download xarray==0.16.0
!pip install arviz

!pip install pymc3==3.8



# !pip download arviz

# !pip download pymc3==3.8
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder





# import arviz

import pymc3 as pm
pm.__version__
exclude_test_patient_data_from_trainset = True



# train = pd.read_csv('/content/drive/My Drive/Kaggle/OSIC/data_osic/train.csv')

# df_train = pd.read_csv('/content/drive/My Drive/Kaggle/OSIC/data_osic/train.csv')

df_train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')



latents = pd.read_csv('../input/latent-features/latent_features.csv')



mu_log_var = pd.read_csv('../input/mu-log-varcsv/mu_log_var.csv')

mu_log_var.columns = ['Patient', 

                      'mu0', 'mu1', 'mu2', 'mu3', 'mu4', 

                      'mu5', 'mu6', 'mu7', 'mu8', 'mu9',

                      'sig0', 'sig1', 'sig2', 'sig3', 'sig4',

                      'sig5', 'sig6', 'sig7', 'sig8', 'sig9']



train_temp = pd.merge(df_train, latents,on='Patient',how='left')

train = pd.merge(train_temp, mu_log_var, on='Patient', how='left')



# test = pd.read_csv('/content/drive/My Drive/Kaggle/OSIC/data_osic/test.csv')

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



# if exclude_test_patient_data_from_trainset:

#     train = train[~train['Patient'].isin(test['Patient'].unique())]



# train = pd.concat([train, test], axis=0, ignore_index=True)\

#     .drop_duplicates()







le_id = LabelEncoder()

train['PatientID'] = le_id.fit_transform(train['Patient'])



train.head()
train['Male'] = train['Sex'].apply(lambda x: 1 if x == 'Male' else 0)



train["SmokingStatus"] = train["SmokingStatus"].astype(

    pd.CategoricalDtype(['Ex-smoker', 'Never smoked', 'Currently smokes'])

)

aux = pd.get_dummies(train["SmokingStatus"], prefix='ss')

aux.columns = ['ExSmoker', 'NeverSmoked', 'CurrentlySmokes']

train['ExSmoker'] = aux['ExSmoker']

train['CurrentlySmokes'] = aux['CurrentlySmokes']



aux = train[['Patient', 'Weeks', 'Percent']].sort_values(by=['Patient', 'Weeks'])

aux = train.groupby('Patient').head(1)

aux = aux.rename(columns={'Percent': 'Percent_base'})

train = pd.merge(train, aux[['Patient', 'Percent_base']], how='left',

                 on='Patient')



train.head()



train['FVC_ref']=round(train['FVC']/(train['Percent']/100))
le_id = LabelEncoder()

train['PatientID'] = le_id.fit_transform(train['Patient'])
train['Male'] = train['Sex'].apply(lambda x: 1 if x == 'Male' else 0)



train["SmokingStatus"] = train["SmokingStatus"].astype(

    pd.CategoricalDtype(['Ex-smoker', 'Never smoked', 'Currently smokes'])

)

aux = pd.get_dummies(train["SmokingStatus"], prefix='ss')

aux.columns = ['ExSmoker', 'NeverSmoked', 'CurrentlySmokes']

train['ExSmoker'] = aux['ExSmoker']

train['CurrentlySmokes'] = aux['CurrentlySmokes']



aux = train[['Patient', 'Weeks', 'Percent']].sort_values(by=['Patient', 'Weeks'])

aux = train.groupby('Patient').head(1)

aux = aux.rename(columns={'Percent': 'Percent_base'})

train = pd.merge(train, aux[['Patient', 'Percent_base']], how='left',

                 on='Patient')



train.head()



train['FVC_ref']=round(train['FVC']/(train['Percent']/100))
n_patients = train['Patient'].nunique()

FVC_obs = train['FVC'].values

Weeks = train['Weeks'].values

PatientID = train['PatientID'].values



X = train[['Weeks', 'Male', 'ExSmoker', 'CurrentlySmokes', 

           'Percent_base_x', '0', '1', '2', '3', '4', '5']].values

#            , '6', '7', '8', '9']].values



PatientID = train['PatientID'].values



with pm.Model() as model_c:

    # create shared variables that can be changed later on

    FVC_obs_shared = pm.Data("FVC_obs_shared", FVC_obs)

    X_shared = pm.Data('X_shared', X)

    PatientID_shared = pm.Data('PatientID_shared', PatientID)

    

    mu_a = pm.Normal('mu_a', mu=1700, sigma=400)

    sigma_a = pm.HalfNormal('sigma_a', 1000.)

    mu_b = pm.Normal('mu_b', mu=-4., sigma=1., shape=X.shape[1])

    sigma_b = pm.HalfNormal('sigma_b', 5.)



    a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_patients)

    b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=(n_patients, X.shape[1]))



    # Model error

    sigma = pm.HalfNormal('sigma', 150.)



    FVC_est = a[PatientID_shared] + (b[PatientID_shared] * X_shared).sum(axis=1)



    # Data likelihood

    FVC_like = pm.Normal('FVC_like', mu=FVC_est,

                         sigma=sigma, observed=FVC_obs_shared)
with model_c:

    trace_c = pm.sample(2000, tune=2000, target_accept=.90, init="adapt_diag")
aux = train.groupby('Patient').first().reset_index()

pred_template = []

for i in range(train['Patient'].nunique()):

    df = pd.DataFrame(columns=['PatientID', 'Weeks'])

    df['Weeks'] = np.arange(-12, 134)

    df['PatientID'] = i

    df['Male'] = aux[aux['PatientID'] == i]['Male'].values[0]

    df['ExSmoker'] = aux[aux['PatientID'] == i]['ExSmoker'].values[0]

    df['CurrentlySmokes'] = aux[aux['PatientID'] == i]['CurrentlySmokes'].values[0]

    df['Percent_base'] = aux[aux['PatientID'] == i]['Percent_base_x'].values[0]



    df['0'] = aux[aux['PatientID'] == i]['0'].values[0]

    df['1'] = aux[aux['PatientID'] == i]['1'].values[0]

    df['2'] = aux[aux['PatientID'] == i]['2'].values[0]

    df['3'] = aux[aux['PatientID'] == i]['3'].values[0]

    df['4'] = aux[aux['PatientID'] == i]['4'].values[0]

    df['5'] = aux[aux['PatientID'] == i]['5'].values[0]

#     df['6'] = aux[aux['PatientID'] == i]['6'].values[0]

#     df['7'] = aux[aux['PatientID'] == i]['7'].values[0]

#     df['8'] = aux[aux['PatientID'] == i]['8'].values[0]

#     df['9'] = aux[aux['PatientID'] == i]['9'].values[0]



    # df['FVC_ref'] = aux[aux['PatientID'] == i]['FVC_ref'].values[0]

    pred_template.append(df)

pred_template = pd.concat(pred_template, ignore_index=True)
# predict posteriors

with model_c:

    pm.set_data({

        "PatientID_shared": pred_template['PatientID'].values.astype(int),

        "X_shared": pred_template[['Weeks', 'Male', 'ExSmoker', 

                                   'CurrentlySmokes', 

                                   'Percent_base','0', '1', '2', '3', '4', '5']].values.astype(int),

#                                    , '6', '7', '8', '9']].values.astype(int),

                 

        "FVC_obs_shared": np.zeros(len(pred_template)).astype(int),

    })

    post_pred = pm.sample_posterior_predictive(trace_c)

    

df = pd.DataFrame(columns=['Patient', 'Weeks', 'FVC_pred', 'sigma'])

df['Patient'] = le_id.inverse_transform(pred_template['PatientID'])

df['Weeks'] = pred_template['Weeks']

df['FVC_pred'] = post_pred['FVC_like'].T.mean(axis=1)

df['sigma'] = post_pred['FVC_like'].T.std(axis=1)

df['FVC_inf'] = df['FVC_pred'] - df['sigma']

df['FVC_sup'] = df['FVC_pred'] + df['sigma']

df = pd.merge(df, train[['Patient', 'Weeks', 'FVC']], how='left', on=['Patient', 'Weeks'])

df = df.rename(columns={'FVC': 'FVC_true'})

df.head()
df = pd.DataFrame(columns=['Patient', 'Weeks', 'Patient_Week', 'FVC', 'Confidence'])

df['Patient'] = pred_template['PatientID']

df['Weeks'] = pred_template['Weeks']

df['Patient_Week'] = df['Patient'].astype(str) + '_' + df['Weeks'].astype(str)

df['FVC'] = post_pred['FVC_like'].T.mean(axis=1)

df['Confidence'] = post_pred['FVC_like'].T.std(axis=1)

final = df[['Patient_Week', 'FVC', 'Confidence']]



final.to_csv('submission.csv', index=False)



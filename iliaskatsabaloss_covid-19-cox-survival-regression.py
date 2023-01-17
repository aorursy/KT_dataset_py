# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/coronavirusdataset/PatientInfo.csv')
! pip install lifelines
from datetime import datetime
from lifelines import CoxPHFitter
from lifelines.utils.sklearn_adapter import sklearn_adapter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import re
import math
sns.set(style="ticks", context="talk")
plt.style.use("dark_background")
pd.set_option('mode.chained_assignment', None)

df = data
df['start_date'] = df.confirmed_date
df['start_date'] = pd.to_datetime(df['start_date'])
df['deceased_date'] = pd.to_datetime(df['deceased_date'])
df['released_date'] = pd.to_datetime(df['released_date'])
# The assumption here is that if the patient is not deceased or recovered, he is censored, meaning we do not know the outcome yet.
df['survived_date'] = df.deceased_date.fillna(df.released_date).fillna(datetime.today())

df = df.drop(df[(df.state == 'deceased') & (df.deceased_date.isna())].index)
df = df.drop(df[(df.state == 'released') & (df.released_date.isna())].index)

df['died'] = df.deceased_date.notnull().astype(int)
df['recovered'] = df.released_date.notnull().astype(int)
df['survived'] = (df.survived_date - df.start_date).dt.days

df['is_male'] = (df.sex == 'male').astype(int)
df['age_temp_a'] = df.age.apply(lambda x: int(re.findall("\d+", str(x))[0]) / 10 if pd.notnull(x) else None)
df['age_temp_b'] = df.birth_year.apply(lambda x: np.floor((2020 - x) / 10) if x != None else None)
df['age_decade'] = df.age_temp_a.fillna(df.age_temp_b)
df = df.drop(df[df.age_decade.isna()].index)
df = df[['patient_id', 'age_decade', 'is_male', 'died', 'recovered', 'survived']]
df = df.drop(df[df.isnull().any(axis = 1)].index)
df = df.drop(df[df.patient_id.duplicated()].index)
df.head()
train, test = train_test_split(df, test_size = 0.3)
X_train_death = train[['died', 'age_decade', 'is_male']]
y_train_death = train['survived']
X_test_death = test[['died', 'age_decade', 'is_male']]
y_test_death = test['survived']

X_train_recovery = train[['recovered', 'age_decade', 'is_male']]
y_train_recovery = train['survived']
X_test_recovery = test[['recovered', 'age_decade', 'is_male']]
y_test_recovery = test['survived']
print('train size : {}, test_size : {}'.format(train.shape[0], test.shape[0]))
def cox_ph_fitter(X_train, y_train, X_test, y_test, event):
    CoxRegression = sklearn_adapter(CoxPHFitter, event_col = event)
    cph = CoxRegression()
    cph.fit(X_train, y_train)
    cph.lifelines_model.print_summary()
    print('---')
    print('Test Score = {}'.format(cph.score(X_test, y_test)))
    return cph.lifelines_model
cph_death = cox_ph_fitter(X_train_death, y_train_death, X_test_death, y_test_death,  'died')
cph_recovery = cox_ph_fitter(X_train_recovery, y_train_recovery, X_test_recovery, y_test_recovery, 'recovered')
base_case_death = pd.Series({'age_decade' : 0, 'is_male' : 0})
cummulative_baseline_hazard_death = cph_death.predict_cumulative_hazard(base_case_death)
baseline_hazard_death = cummulative_baseline_hazard_death.diff().fillna(cummulative_baseline_hazard_death)

base_case_recovery = pd.Series({'age_decade' : 0, 'is_male' : 0})
cummulative_baseline_hazard_recovery = cph_recovery.predict_cumulative_hazard(base_case_recovery)
baseline_hazard_recovery = cummulative_baseline_hazard_recovery.diff().fillna(cummulative_baseline_hazard_recovery)

baseline_hazards = pd.merge(baseline_hazard_death[0:], baseline_hazard_recovery[0:], left_index=True, right_index=True)
baseline_hazards = baseline_hazards.rename(columns = {'0_x' : 'baseline_hazard_death', '0_y' : 'baseline_hazard_recovery'})
baseline_hazards.head()
test['key'] = 1
baseline_hazards['key'] = 1

#cross join of patient in the test set with the baseline hazards
test_lines = pd.merge(test, baseline_hazards[:30], on = 'key')

#simple row number, indicating the number of days since confirmed.
test_lines['days'] = test_lines.groupby('patient_id').cumcount()

#according to the definition of the hazard function in cox regression, we calculate the hazard for each patient individually
#in effect this is the conditional probability of death and recovery for each patient, given the patient has survived so far
test_lines['p_death_given_s'] = test_lines.baseline_hazard_death * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_death.params_))
test_lines['p_recovery_given_s'] = test_lines.baseline_hazard_recovery * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_recovery.params_))

test_lines['survive_death'] = 1-test_lines.p_death_given_s
test_lines['survive_recovery'] = 1-test_lines.p_recovery_given_s

#probability of survival defined as if the patient did not die and did not recover so far 
test_lines['p_survive'] = test_lines.groupby('patient_id').survive_death.cumprod() * test_lines.groupby('patient_id').survive_recovery.cumprod()

#unconditional probability of death and recovery. 
test_lines['p_death'] = test_lines.p_death_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)
test_lines['p_recovery'] = test_lines.p_recovery_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)

#observed deaths and recoveries
test_lines['recovered'] = ((test_lines.survived == test_lines.days) & (test_lines.recovered == 1)).astype(int)
test_lines['died'] = ((test_lines.survived == test_lines.days) & (test_lines.died == 1)).astype(int)
agg = test_lines.groupby('days').agg({
    'died' : np.sum
    ,'recovered' : np.sum
    ,'p_death' : np.sum
    ,'p_recovery' : np.sum
    ,'patient_id' : pd.Series.nunique
}).reset_index()
agg['death_rate'] = agg.died.cumsum() / agg.patient_id
agg['death_rate_pred'] = agg.p_death.cumsum() / agg.patient_id
agg['recovery_rate'] = agg.recovered.cumsum() / agg.patient_id
agg['recovery_rate_pred'] = agg.p_recovery.cumsum() / agg.patient_id
sns.lineplot(agg.days, agg.death_rate_pred, label = 'predictions')
sns.lineplot(agg.days, agg.death_rate, label = 'actuals')
sns.lineplot(agg.days, agg.recovery_rate_pred, label = 'predictions')
sns.lineplot(agg.days, agg.recovery_rate, label = 'actuals')
def cox_ph_fitter(X_train, y_train, X_test, y_test, event):
    CoxRegression = sklearn_adapter(CoxPHFitter, event_col = event)
    cph = CoxRegression()
    cph.fit(X_train, y_train)
    #cph.lifelines_model.print_summary()
    #print('---')
    #print('Test Score = {}'.format(cph.score(X_test, y_test)))
    return cph.lifelines_model

aggregations = pd.DataFrame(columns = ['iteration','days', 'death_rate', 'death_rate_pred', 'recovery_rate', 'recovery_rate_pred'])

for iteration in range (1,21):
    train, test = train_test_split(df, test_size = 0.3)
    X_train_death = train[['died', 'age_decade', 'is_male']]
    y_train_death = train['survived']
    X_test_death = test[['died', 'age_decade', 'is_male']]
    y_test_death = test['survived']

    X_train_recovery = train[['recovered', 'age_decade', 'is_male']]
    y_train_recovery = train['survived']
    X_test_recovery = test[['recovered', 'age_decade', 'is_male']]
    y_test_recovery = test['survived']

    cph_death = cox_ph_fitter(X_train_death, y_train_death, X_test_death, y_test_death,  'died')
    cph_recovery = cox_ph_fitter(X_train_recovery, y_train_recovery, X_test_recovery, y_test_recovery, 'recovered')

    base_case_death = pd.Series({'age_decade' : 0, 'is_male' : 0})
    cummulative_baseline_hazard_death = cph_death.predict_cumulative_hazard(base_case_death)
    baseline_hazard_death = cummulative_baseline_hazard_death.diff().fillna(cummulative_baseline_hazard_death)

    base_case_recovery = pd.Series({'age_decade' : 0, 'is_male' : 0})
    cummulative_baseline_hazard_recovery = cph_recovery.predict_cumulative_hazard(base_case_recovery)
    baseline_hazard_recovery = cummulative_baseline_hazard_recovery.diff().fillna(cummulative_baseline_hazard_recovery)

    baseline_hazards = pd.merge(baseline_hazard_death[0:], baseline_hazard_recovery[0:], left_index=True, right_index=True)
    baseline_hazards = baseline_hazards.rename(columns = {'0_x' : 'baseline_hazard_death', '0_y' : 'baseline_hazard_recovery'})

    test['key'] = 1
    baseline_hazards['key'] = 1

    #cross join of patient in the test set with the baseline hazards
    test_lines = pd.merge(test, baseline_hazards[:30], on = 'key')

    #simple row number, indicating the number of days since confirmed.
    test_lines['days'] = test_lines.groupby('patient_id').cumcount()

    #according to the definition of the hazard function in cox regression, we calculate the hazard for each patient individually
    #in effect this is the conditional probability of death and recovery for each patient, given the patient has survived so far
    test_lines['p_death_given_s'] = test_lines.baseline_hazard_death * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_death.params_))
    test_lines['p_recovery_given_s'] = test_lines.baseline_hazard_recovery * np.exp(np.dot(test_lines[['age_decade', 'is_male']], cph_recovery.params_))

    test_lines['survive_death'] = 1-test_lines.p_death_given_s
    test_lines['survive_recovery'] = 1-test_lines.p_recovery_given_s

    #probability of survival defined as if the patient did not die and did not recover so far 
    test_lines['p_survive'] = test_lines.groupby('patient_id').survive_death.cumprod() * test_lines.groupby('patient_id').survive_recovery.cumprod()

    #unconditional probability of death and recovery. 
    test_lines['p_death'] = test_lines.p_death_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)
    test_lines['p_recovery'] = test_lines.p_recovery_given_s * test_lines.groupby('patient_id').p_survive.shift().fillna(1)

    #observed deaths and recoveries
    test_lines['recovered'] = ((test_lines.survived == test_lines.days) & (test_lines.recovered == 1)).astype(int)
    test_lines['died'] = ((test_lines.survived == test_lines.days) & (test_lines.died == 1)).astype(int)

    agg = test_lines.groupby('days').agg({
        'died' : np.sum
        ,'recovered' : np.sum
        ,'p_death' : np.sum
        ,'p_recovery' : np.sum
        ,'patient_id' : pd.Series.nunique
    }).reset_index()
    agg['death_rate'] = agg.died.cumsum() / agg.patient_id
    agg['death_rate_pred'] = agg.p_death.cumsum() / agg.patient_id
    agg['recovery_rate'] = agg.recovered.cumsum() / agg.patient_id
    agg['recovery_rate_pred'] = agg.p_recovery.cumsum() / agg.patient_id
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    sns.lineplot(agg.days, agg.death_rate_pred, label = 'predictions', ax = ax1)
    sns.lineplot(agg.days, agg.death_rate, label = 'actuals', ax = ax1)

    sns.lineplot(agg.days, agg.recovery_rate_pred, label = 'predictions', ax = ax2)
    sns.lineplot(agg.days, agg.recovery_rate, label = 'actuals', ax = ax2)
    fig.suptitle('Iteration : {}'.format(iteration) )
    
    agg['iteration'] = iteration
    agg_to_append = agg[['iteration', 'days', 'death_rate', 'death_rate_pred', 'recovery_rate', 'recovery_rate_pred']]
    aggregations = aggregations.append(agg_to_append, ignore_index = True)
    
aggregations['squared_death_diff'] = (aggregations.death_rate - aggregations.death_rate_pred) ** 2
aggregations['squared_recovery_diff'] = (aggregations.recovery_rate - aggregations.recovery_rate_pred) ** 2
errors = aggregations.groupby('days').agg({
    'squared_death_diff' : np.sum
    ,'squared_recovery_diff' : np.sum
    ,'iteration' : np.size
}).reset_index()
errors['RMSD_death_rate'] = np.sqrt(errors.squared_death_diff / errors.iteration)
errors['RMSD_recovery_rate'] = np.sqrt(errors.squared_recovery_diff / errors.iteration)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))
sns.lineplot(errors.days, errors.RMSD_death_rate, ax = ax1)
sns.lineplot(errors.days, errors.RMSD_recovery_rate, ax = ax2)

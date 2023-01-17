import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 20)

pd.set_option('display.max_columns', 200)
df = pd.read_csv('/kaggle/input/cord-19-create-dataframe/cord19_df.csv')
ct_df = pd.read_csv('/kaggle/input/covid-19-international-clinical-trials/data/ClinicalTrials.gov_COVID_19.csv')



ct_df.shape, ct_df.columns
ict_df = pd.read_csv('/kaggle/input/covid-19-international-clinical-trials/data/ICTRP_COVID_19.csv')



ict_df.shape, ict_df.columns
ict_df.head(3)
ict = ict_df[['TrialID', 'web address', 'Study type', 'Study design', 'Intervention', 'Primary outcome']]

ct = ct_df[['NCT Number', 'URL', 'Study Type', 'Study Designs', 'Interventions', 'Outcome Measures']]
ict.columns = ['id', 'url', 'study_type', 'study_design', 'intervention', 'outcome']

ct.columns = ['id', 'url', 'study_type', 'study_design', 'intervention', 'outcome']
all_trials = ict.append(ct, ignore_index=True)
all_trials.head()
# all_trials[all_trials.id.duplicated(keep=False)].sort_values('id').head()
all_trials.drop_duplicates(subset='id', keep='last', inplace=True)
all_trials.shape
all_trials.id.str[:6].value_counts()
all_trials.id
all_trials.id[all_trials.id.str.startswith('EUCTR')] # use this to see the patterns
reg_nct = 'NCT[0-9]{8}'

reg_chi = 'ChiCTR[0-9]{10}'

reg_eu = 'EUCTR[0-9]{4}-[0-9]{6}-[0-9]{2}-[A-Z]{2}'

reg_ir = 'IRCT[0-9]+N[0-9]{1,2}'

reg_isrctn = 'ISRCTN[0-9]{8}'

reg_jprn = 'JPRN-[0-9a-zA-Z]+'

reg_tctr = 'TCTR[0-9]{11}'

reg_actrn = 'ACTRN[0-9]{14}'

reg_drks = 'DRKS[0-9]{8}'



registries = [reg_nct, reg_chi, reg_eu, reg_ir, reg_isrctn, reg_jprn, reg_tctr, reg_actrn, reg_drks]



reg = ('|').join(registries)

reg = r'({})'.format(reg)



reg
pd.Series(['The trial has been registered in Chinese Clinical Trial Registry (ChiCTR2000029981).']).str.extract(reg)
len(all_trials), len(all_trials.id.str.extract(reg))
trials = (df.title.fillna('') + ' ' + df.abstract.fillna('') + ' ' + df.body_text.fillna('')).str.extract(reg)
df['trial_id'] = trials
trials.notnull().sum()
# df[df.trial_id.notnull() & df.is_covid19].shape

# Mike Honey edit - keep non-COVID-19 papers

df[df.trial_id.notnull()].shape
# final = pd.merge(left=df, right=all_trials, left_on='trial_id', right_on='id', how='left', suffixes=(None, '_trial'))
final = pd.merge(left=df[['paper_id', 'trial_id']], right=all_trials, left_on='trial_id', right_on='id', how='inner').drop(columns=['id'])
final.tail()
final.shape
final.trial_id.nunique()
df.is_covid19.sum()
final.to_csv('trial_info.csv', index=False)
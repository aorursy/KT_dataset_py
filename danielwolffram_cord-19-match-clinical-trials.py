import numpy as np

import pandas as pd
pd.set_option('display.max_rows', 20)

pd.set_option('display.max_columns', 200)
df = pd.read_csv('/kaggle/input/cord-19-create-dataframe-may-12-update/cord19_df.csv')
df.head()
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
all_trials.id.duplicated().sum()
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

reg_nl = 'NL[0-9]{4}'



registries = [reg_nct, reg_chi, reg_eu, reg_ir, reg_isrctn, reg_jprn, reg_tctr, reg_actrn, reg_drks, reg_nl]



reg = ('|').join(registries)

reg = r'({})'.format(reg)



reg
pd.Series(['The trial has been registered in Chinese Clinical Trial Registry (ChiCTR2000029981). blabla NCT04275245', 'NCT04275245']).str.findall(reg)
all_trials.id.str.extract(reg)[0]
(all_trials.id == all_trials.id.str.extract(reg)[0]).sum()
trials = (df.title.fillna('') + ' ' + df.abstract.fillna('') + ' ' + df.body_text.fillna('')).str.findall(reg)
df['trial_id'] = trials.apply(lambda x: list(dict.fromkeys(x))) # remove multiple occurences
sum([len(x)!=0 for x in trials])
# trials.notnull().sum()
df[[len(x)!=0 for x in trials] & df.is_covid19].shape
all_trials.set_index('id', inplace=True)
trials_df = pd.DataFrame(trials, columns=['trials'])
trials_df.trials = trials_df.trials.apply(lambda x: [i for i in pd.Index(x).intersection(all_trials.index)])
trials_df.set_index(df.cord_uid, inplace=True)
trials_df = trials_df[trials_df.trials.str.len() != 0]
all_trials.loc[all_trials.index.intersection(['NCT04276688', 'NCT04319172', 'NCT04319172'])].url.values.tolist()
def get_urls(df, trial_ids):

    idx = pd.Index(trial_ids)

    if len(idx) > 0:

        return df.loc[idx].url.values.tolist()

    else:

        return None
trials_df['trial_url'] = trials_df.trials.apply(lambda x: get_urls(all_trials, x))
# trials_df = trials_df.dropna(subset=['trial_url'])
trials_df.loc['00s3wgem']
trials_df.head()
trials_df.shape
len(df.set_index('cord_uid')[df.set_index('cord_uid').is_covid19==True].index.intersection(trials_df.index))
df.set_index('cord_uid').loc['00s3wgem'].url
df.is_covid19.sum()
trials_df
trials_df.to_csv('trial_urls.csv')
a = pd.read_csv('trial_urls.csv', index_col='cord_uid' , converters={'trial_url': eval})
a.head()
len(trials_df.explode(column='trials')), len(trials_df.explode(column='trial_url'))
trials_exploded = trials_df.explode(column='trials')



trials_exploded.trial_url = trials_df.explode(column='trial_url').trial_url.values



trials_exploded.head()
trials_exploded.to_csv('trials_exploded.csv')
# def get_trials(df, trial_ids):

#     idx = df.index.intersection(trial_ids)

#     if len(idx) > 0:

#         return df.loc[idx]

#     else:

#         return None



# trials_df['trial_info'] = trials_df.trials.apply(lambda x: get_trials(all_trials, x))



# trials_df = trials_df.set_index(df.paper_id).dropna(subset=['trial_info'])



# trials_df.loc['188e7ff1e260864c89f266b5597de26d69a84660'].trial_info



# trials_df.shape



# trials_df.to_csv('trials_df.csv')



# a = pd.read_csv('trials_df.csv', index_col='paper_id', converters={'trial_info': eval}) # how to load this?
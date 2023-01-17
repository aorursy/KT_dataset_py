# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import seaborn as sns


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
df_train = pd.read_csv("../input/lish-moa/train_features.csv")
df_test = pd.read_csv("../input/lish-moa/test_features.csv")
df_train_targets_nonscored = pd.read_csv("../input/lish-moa/train_targets_nonscored.csv")
df_train_targets_scored = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
df_sample_submission = pd.read_csv("../input/lish-moa/sample_submission.csv")
pd.set_option('display.max_columns',500)
df_train.head()
df_train.isnull().sum().nlargest()
df_train.info()
df_train.select_dtypes(include=['object']).dtypes
df_train.drop('sig_id',axis=1,inplace=True)
df_train['cp_dose'].value_counts()/len(df_train)
df_train['dataset'] = 'train'
df_test['dataset'] = 'test'

df = pd.concat([df_train, df_test])
sns.countplot(x = 'cp_dose',hue='dataset',data=df)
df_train['cp_type'].value_counts()/len(df_train)
sns.countplot(x = 'cp_type',hue='dataset',data=df)
sns.countplot(x = 'cp_time',hue='dataset',data=df)
gs = df_train[:1][[col for col in df_train.columns if 'g-' in col]].values.reshape(-1,1)
import matplotlib.pyplot as plt
plt.plot(gs)

plt.plot(sorted(gs))
df_train['g-0'].plot(kind='hist')
df_train['c-0'].plot(kind='hist')
correlated_variables = []
feature_columns = list(df_train.select_dtypes(exclude=['object']).columns)
for i in range(0,len(feature_columns)):
    for j in range(i+1,len(feature_columns)):
        #print(col1,col2)
        if abs(df_train[feature_columns[i]].corr(df_train[feature_columns[j]])) > 0.9:
            correlated_variables.append(feature_columns[i])
            correlated_variables.append(feature_columns[j])
correlated_variables = set(correlated_variables)
plt.figure(figsize=(20,10))
sns.heatmap(df[correlated_variables].corr())

df_train._get_numeric_data().describe()

sns.boxplot(df_train['cp_type'],df_train['g-30'])
df_train_targets_scored.head()
df_train_targets_scored.columns
df_train_targets_scored.isnull().sum().nlargest()
df_train_targets_scored.drop('sig_id',axis=1,inplace=True)
len(df_train_targets_scored.columns)
df_train_targets_scored.sum(axis=1).nlargest()
plt.figure(figsize=(15,15))
data = df_train_targets_scored.sum()
data_largest = data.nlargest(50)
sns.barplot(data_largest.values,data_largest.index,orient='h')
data_smallest=data.nsmallest(50)
plt.figure(figsize=(15,15))
sns.barplot(data_smallest.values,data_smallest.index,orient='h')
len(df_train_targets_scored.columns)
x = df_train_targets_scored.sum().sort_values(ascending=False).reset_index()

x.columns = ['column', 'count']
x['count'] = x['count'] * 100 / len(df_train_targets_scored)
plt.figure(figsize=(10,30))
sns.barplot(x['count'],x['column'],orient='h')
y = df_train_targets_scored.sum(axis=1).reset_index()
y.columns = ['records','total_labels']
y.head()
sns.countplot(y['total_labels'])
y['total_labels'].value_counts()
target_columns = list(df_train_targets_scored.columns)
feature_target_corr_df = pd.DataFrame()
for target_col in target_columns:
    corr_list = []
    for feature_col in feature_columns:
        corr = df_train[feature_col].corr(df_train_targets_scored[target_col])
        corr_list.append(corr)
    feature_target_corr_df[target_col] = corr_list
    
feature_target_corr_df['train_features'] = feature_columns
feature_target_corr_df = feature_target_corr_df.set_index('train_features')
feature_target_corr_df.head()
maxCol = lambda x : max(x.max(),x.min(),key=abs)
high_scores=feature_target_corr_df.apply(maxCol, axis=0).reset_index()
high_scores.columns = ['column','best_correlation']
#x

plt.figure(figsize=(30,10))

g = sns.barplot(high_scores['column'],high_scores['best_correlation'])
g.set_xticklabels(g.get_xticklabels(),rotation=90)
plt.show()
# x['max_value']
col_df = pd.DataFrame()
tr_cols = list()
tar_cols = list()
for col in feature_target_corr_df.columns:
    tar_cols.append(col)
    tr_cols.append(feature_target_corr_df[col].abs().sort_values(ascending=False).reset_index()['train_features'].head(1).values[0])

col_df['column'] = tar_cols
col_df['train_best_column'] = tr_cols

total_scores = pd.merge(high_scores, col_df)
total_scores
training_col_high_corr = total_scores['train_best_column'].value_counts().reset_index()
training_col_high_corr.columns = ['train_columns','count']

plt.figure(figsize=(10,10))
sns.barplot(training_col_high_corr['count'].head(33),training_col_high_corr['train_columns'].head(33),orient='h')
last_term_target_column = dict()
for col in target_columns:
    try:
        last_term_target_column[col.split("_")[-1]] += 1
    except:
        last_term_target_column[col.split("_")[-1]] = 1
target_column_group_df = pd.DataFrame(last_term_target_column.items(),columns = ['group','count'])
target_column_group_df.sort_values('count',ascending=False)
target_column_group_df = target_column_group_df[target_column_group_df['count']>1]
target_column_group_df.head()
sns.barplot(target_column_group_df['group'],target_column_group_df['count'])
answer = list()
for group in target_column_group_df.group.tolist():
    agent_list = list()
    for item in target_columns:
        if item.split('_')[-1] == group:
            agent_list.append(item)
    print(agent_list)
#     agent_df = train_target[agent_list]
#     data = agent_df.astype(bool).sum(axis=1).reset_index()
#     answer.append(data[0].max())
agent_df = df_train_targets_scored[['5-alpha_reductase_inhibitor', '11-beta-hsd1_inhibitor', 'acat_inhibitor', 'acetylcholinesterase_inhibitor', 'akt_inhibitor', 'aldehyde_dehydrogenase_inhibitor', 'alk_inhibitor', 'angiogenesis_inhibitor', 'aromatase_inhibitor', 'atm_kinase_inhibitor', 'atp_synthase_inhibitor', 'atpase_inhibitor', 'atr_kinase_inhibitor', 'aurora_kinase_inhibitor', 'autotaxin_inhibitor', 'bacterial_30s_ribosomal_subunit_inhibitor', 'bacterial_50s_ribosomal_subunit_inhibitor', 'bacterial_cell_wall_synthesis_inhibitor', 'bacterial_dna_gyrase_inhibitor', 'bacterial_dna_inhibitor', 'bacterial_membrane_integrity_inhibitor', 'bcl_inhibitor', 'bcr-abl_inhibitor', 'beta_amyloid_inhibitor', 'bromodomain_inhibitor', 'btk_inhibitor', 'calcineurin_inhibitor', 'carbonic_anhydrase_inhibitor', 'casein_kinase_inhibitor', 'catechol_o_methyltransferase_inhibitor', 'cdk_inhibitor', 'chk_inhibitor', 'cholesterol_inhibitor', 'coagulation_factor_inhibitor', 'cyclooxygenase_inhibitor', 'cytochrome_p450_inhibitor', 'dihydrofolate_reductase_inhibitor', 'dipeptidyl_peptidase_inhibitor', 'dna_inhibitor', 'egfr_inhibitor', 'elastase_inhibitor', 'erbb2_inhibitor', 'faah_inhibitor', 'farnesyltransferase_inhibitor', 'fgfr_inhibitor', 'flt3_inhibitor', 'focal_adhesion_kinase_inhibitor', 'fungal_squalene_epoxidase_inhibitor', 'gamma_secretase_inhibitor', 'glutamate_inhibitor', 'gsk_inhibitor', 'hcv_inhibitor', 'hdac_inhibitor', 'histone_lysine_demethylase_inhibitor', 'histone_lysine_methyltransferase_inhibitor', 'hiv_inhibitor', 'hmgcr_inhibitor', 'hsp_inhibitor', 'igf-1_inhibitor', 'ikk_inhibitor', 'integrin_inhibitor', 'jak_inhibitor', 'kit_inhibitor', 'leukotriene_inhibitor', 'lipase_inhibitor', 'lipoxygenase_inhibitor', 'mdm_inhibitor', 'mek_inhibitor', 'membrane_integrity_inhibitor', 'monoacylglycerol_lipase_inhibitor', 'monoamine_oxidase_inhibitor', 'monopolar_spindle_1_kinase_inhibitor', 'mtor_inhibitor', 'nfkb_inhibitor', 'nitric_oxide_production_inhibitor', 'nitric_oxide_synthase_inhibitor', 'norepinephrine_reuptake_inhibitor', 'p38_mapk_inhibitor', 'p-glycoprotein_inhibitor', 'parp_inhibitor', 'pdgfr_inhibitor', 'pdk_inhibitor', 'phosphodiesterase_inhibitor', 'phospholipase_inhibitor', 'pi3k_inhibitor', 'pkc_inhibitor', 'prostaglandin_inhibitor', 'proteasome_inhibitor', 'protein_kinase_inhibitor', 'protein_phosphatase_inhibitor', 'protein_synthesis_inhibitor', 'protein_tyrosine_kinase_inhibitor', 'raf_inhibitor', 'ras_gtpase_inhibitor', 'rho_associated_kinase_inhibitor', 'ribonucleoside_reductase_inhibitor', 'rna_polymerase_inhibitor', 'serotonin_reuptake_inhibitor', 'sodium_channel_inhibitor', 'src_inhibitor', 'syk_inhibitor', 'tgf-beta_receptor_inhibitor', 'thrombin_inhibitor', 'thymidylate_synthase_inhibitor', 'tnf_inhibitor', 'topoisomerase_inhibitor', 'tropomyosin_receptor_kinase_inhibitor', 'tubulin_inhibitor', 'tyrosine_kinase_inhibitor', 'ubiquitin_specific_protease_inhibitor', 'vegfr_inhibitor', 'wnt_inhibitor']]
agent_df.astype(bool).sum(axis=1).reset_index()[0].max()

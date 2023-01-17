import pandas as pd
df_1 = pd.read_csv('../input/rsna2019-csv/stage_1_sample_submission.csv')

df_2 = pd.read_csv('../input/rsna2019-csv/stage_2_train.csv')
set_df_1_id = set(df_1['ID'])

idx         = [True if id_tmp in set_df_1_id else False for id_tmp in df_2['ID']]
df_2[idx]['Label'].values.reshape(-1, 6).sum(axis=0)
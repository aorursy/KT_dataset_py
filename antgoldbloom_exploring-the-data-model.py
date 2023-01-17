import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_public_regattas = pd.read_csv('../input/public_regatta.tsv',sep='\t')

df_public_regattas.head()
df_team = pd.read_csv('../input/team.tsv',sep='\t')

df_team.head()
df_attendee = pd.read_csv('../input/attendee.tsv',sep='\t')

df_attendee.head()
df_active_sailor = pd.read_csv('../input/active_sailor.tsv',sep='\t')

df_active_sailor.head()
df_race = pd.read_csv('../input/race.tsv',sep='\t')

df_race.head()
df_finishes = pd.read_csv('../input/finish.tsv',sep='\t')

df_finishes.head()
df_sailor_rank = pd.merge(df_team,df_attendee,left_on='id',right_on='team',how='innner')

df_sailor_rank[['sailor','dt_rank','regatta','school']]
df_sailor_rank_with_names = pd.merge(df_sailor_rank,df_active_sailor,left_on='sailor',right_on='id',how='outer')

df_sailor_rank_with_names['school'] = df_sailor_rank_with_names['school_x'] 

df_sailor_rank_with_names[['sailor','dt_rank','regatta','school','first_name','last_name']]
df_dt_rp = pd.read_csv('../input/dt_rp.tsv',sep='\t')

df_team_division = pd.read_csv('../input/dt_team_division.tsv',sep='\t')

#the ID column is getting read in as a strong, which means causes an issue for joining to dt_rp. Convert to strong

df_team_division['id'] = pd.to_numeric(df_team_division['id'],errors='coerce')

df_team_division_results = pd.merge(df_dt_rp,df_team_division,left_on='team_division',right_on='id',how='inner')

df_team_division_results_with_team = pd.merge(df_team_division_results,df_team,left_on='team',right_on='id',how='inner')

df_team_division_results_with_team['rank'] = df_team_division_results_with_team['rank_x']

df_team_division_results_with_team[['sailor','rank','school','regatta']]


df_team_division_results
#df_team_division

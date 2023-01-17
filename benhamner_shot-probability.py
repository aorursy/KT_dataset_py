import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/data.csv',header = 0);
df_train = df.dropna()
index = df['shot_made_flag'].apply(np.isnan)
df_test = df[index]
df_train.head(5)
#df_test.head(10)
""" Prob of action_type + combined_shot_type going in"""
tup = [df_train['action_type'],df_train['combined_shot_type'],df_train['shot_made_flag']]
tup = pd.concat(tup,axis = 1)
#df_new = pd.DataFrame(columns=['action_type','combined_shot_type','probability_success'])
grouped = tup.groupby(['action_type','combined_shot_type']).agg([np.sum,'count']).reset_index()
prob =  grouped['shot_made_flag']['sum']/grouped['shot_made_flag']['count']
a = {'action_type':grouped.action_type,'combined_shot_type':grouped.combined_shot_type,'prob':prob}
df = pd.DataFrame(a)
df.sort(['prob'], ascending=[False], inplace=True)
sns.factorplot(x='prob', y='action_type',hue = 'combined_shot_type', data=df, size=12)
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
pd.__version__
file_name = "_election_results_by_pcon.csv"
raw_df_2015 = pd.read_csv('/kaggle/input/' + '2015'+ file_name,header=0, thousands=',' , quotechar = '"');

raw_df_2017 = pd.read_csv('/kaggle/input/' + '2017' + file_name,header=0, thousands=',' , quotechar = '"');
raw_df_2015.fillna(0, inplace=True);

raw_df_2017.fillna(0, inplace=True);
raw_df_2015['Year']=2015

raw_df_2017['Year']=2017
raw_df_2015.set_index('id', inplace=True);

raw_df_2017.set_index('id', inplace=True);

raw_df = pd.concat([raw_df_2015,raw_df_2017], sort=False).sort_index()

raw_df.head()
parties = raw_df.columns[5:18]

cols = list(['Constituency','Year']) + list(parties)

print(parties)

print(cols)
df = raw_df[cols].set_index('Year',append=True)

df.head()
#check if there is any new constituency or costituency dropping

s = df.groupby('id')['Conservative'].count()

s = s[s < 2]

s

df[df.index.get_level_values('id').isin(s.index)]

df_2017=df.xs(2017,level=1,drop_level=True)[parties]

df_2017['winner'] = df_2017.idxmax(axis=1)

df_2017.groupby(['winner'])['Conservative'].count()
df_2015=df.xs(2015,level=1,drop_level=True)[parties]

df_2015['winner'] = df_2015.idxmax(axis=1)

df_2015.groupby(['winner'])['Conservative'].count()
df_base=df_2015.drop(['winner'],axis=1);

sLength = len(df_2015['Conservative']);



df_base.head()
#build the percentage moves matrix



sources = ['Labour', 'Conservative'];

zero_s = pd.Series(np.zeros(len(parties)),index = parties)



vote_moves = pd.DataFrame(index = parties, columns = sources);



vote_moves['Labour'] = zero_s;

vote_moves['Conservative'] = zero_s;



#Labour out votes

vote_moves['Labour']['Liberal Democrats'] = 0.1

vote_moves['Labour']['UKIP'] = 0.1

vote_moves['Labour']['SNP'] = 0



#Conservative out votes

vote_moves['Conservative']['Liberal Democrats'] = 0.05 #0.10

vote_moves['Conservative']['UKIP'] = 0.10 #0.10

vote_moves['Conservative']['SNP'] = 0



#normalize

for idx in sources:    

    vote_moves[idx][idx] = - vote_moves[idx].drop(idx).sum()



vote_moves
df_shifts = df_base.copy();

df_shifts[parties] = 0;

for s in sources:

    for idx in parties:

        df_shifts[idx] += df_base[s] * vote_moves[s].loc[idx]





#should be zeros

df_shifts[parties].sum(axis=1)

#overwrite where SNP is present

snp_mask = df_base.loc[:,'SNP']>0

df_shifts[snp_mask] = 0; 
#Labour out votes

#vote_moves = pd.DataFrame(index = parties, columns = ['Labour']);

vote_moves['Labour'].values[:] = 0;



vote_moves['Labour']['SNP'] = 0.3

vote_moves['Labour']['Labour'] = -0.3





vote_moves
for s in vote_moves.columns:

    for idx in vote_moves.index:

        df_shifts[idx][snp_mask] += df_base[s][snp_mask] * vote_moves[s].loc[idx]





#should be zeros

df_shifts[parties][snp_mask]
df_scenario = df_base + df_shifts;

df_scenario.head()
df_scenario['winner'] = df_scenario.idxmax(axis=1)
parl = df_scenario.groupby(['winner'])['Conservative'].count()



print(parl)

print("Total number of seats is {}".format(parl.sum()))
parl[['Conservative', 'UKIP', 'DUP']].sum()
parl[['Labour', 'Liberal Democrats', 'SNP', 'Green']].sum()
tot_votes = df_scenario[parties].sum().sum()

df_scenario[parties].sum() / tot_votes
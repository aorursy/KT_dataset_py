# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/error_matrix.csv').rename(columns={'Unnamed: 0': 'defense'})

df = df.set_index('defense')
def_team = pd.read_csv('../input/defense_results.csv').set_index('KaggleTeamId')

nt_team = pd.read_csv('../input/non_targeted_attack_results.csv').set_index('KaggleTeamId')
from sklearn.cluster import KMeans
err_cols = list(df.columns)

err_cols = err_cols[:err_cols.index('baseline_fgsm')]
kmeans = KMeans(n_clusters=20).fit(df[err_cols])
df['def_cl'] = kmeans.labels_
def_team.loc[df.loc[df['def_cl']==df.loc['820405','def_cl']].index]
print('inceptionv3 teams', (df['def_cl']==df.loc['baseline_inceptionv3', 'def_cl']).sum())
print('inception resnet v2 teams', (df['def_cl']==df.loc['baseline_ens_adv_inception_renset_v2','def_cl']).sum())
df_a = pd.read_csv('../input/error_matrix.csv').rename(columns={'Unnamed: 0': 'attack'}).set_index('attack').transpose()
df_a = df_a.loc[err_cols + ['baseline_fgsm', 'baseline_randnoise', 'baseline_noop']]
kmeans_a = KMeans(n_clusters=20).fit(df_a)
df_a['a_cl'] = kmeans_a.labels_
nt_team.loc[df_a.loc[df_a['a_cl']==df_a.loc['823044','a_cl']].index]
nt_team.loc[df_a.loc[df_a['a_cl']==df_a.loc['875459','a_cl']].index]
a_cols = [c for c in df_a.columns if c is not 'a_cl']
(df_a.loc['828273', a_cols]-df_a.loc['875459', a_cols]).abs().describe()
(df_a.loc['828273', a_cols]-df_a.loc['823044', a_cols]).abs().describe()
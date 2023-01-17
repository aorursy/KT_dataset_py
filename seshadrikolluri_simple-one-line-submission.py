import pandas as pd

pd.merge(pd.read_csv('../input/test.csv'),

         (pd.read_csv('../input/scalar_coupling_contributions.csv')

          .drop(columns = ['atom_index_0','atom_index_1'])

          .groupby(['type']).median().sum(axis = 1)

          .reset_index()

          .rename(columns = {0: 'scalar_coupling_constant'}))

        )[['id','scalar_coupling_constant']].to_csv('baseline_submission.csv',index=False)
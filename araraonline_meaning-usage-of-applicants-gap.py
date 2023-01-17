import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)


# load data

print("Loading data...")
base_df = pd.read_pickle('../input/gapmeaning/schools2017.pkl')

ex1 = pd.read_csv('../input/gapmeaning/expected_testers.csv', index_col=0)
ex2 = pd.read_csv('../input/gapmeaning/expected_testers2.csv', index_col=0)
ex = pd.concat([ex1, ex2])


# generate values

print("Generating values...")
hs_counts = base_df['# Students in HS Admissions']

cnt_shsat_testers = base_df['# SHSAT Testers'].fillna(5)  # schools with up to 5 applicants
pct_shsat_testers = cnt_shsat_testers / hs_counts
ex_cnt_shsat_testers = ex['Expected % of SHSAT Testers'] * hs_counts
ex_pct_shsat_testers = ex['Expected % of SHSAT Testers']

df_cnt_shsat_testers = cnt_shsat_testers - ex_cnt_shsat_testers
df_pct_shsat_testers = pct_shsat_testers - ex_pct_shsat_testers
df_ratio = df_cnt_shsat_testers / cnt_shsat_testers

pct_to_odds = lambda p: p / (1 - p)
chn_multiplier = pct_to_odds(pct_shsat_testers) / pct_to_odds(ex_pct_shsat_testers)


# assemble table

print("Creating table...")
table = pd.DataFrame({
    'Actual Sit #': cnt_shsat_testers,
    'Estimated Sit #': ex_cnt_shsat_testers,
    'Diff Sit #': df_cnt_shsat_testers,
    
    'Actual Sit %': pct_shsat_testers,
    'Estimated Sit %': ex_pct_shsat_testers,
    'Diff Sit %': df_pct_shsat_testers,
    
    'Difference Ratio': df_ratio,
    'Chance Multiplier': chn_multiplier,
})


# join with original table

print("Joining table with schools information...")
joined = base_df.join(table)
joined = joined.drop([
    '# SHSAT Testers',
    '# SHSAT Offers',
    '% SHSAT Testers',
    '% SHSAT Offers',
], axis=1)


# export results

print("Exporting results...")
joined.to_csv('deliver.csv')

print()
print("Done!")
f2_columns = ['Latitude', 'Longitude', 'Economic Need Index',
              'Mean Scale Score - ELA', 'Mean Scale Score - Math',
              'Difference Ratio', 'Chance Multiplier']
f2_columns += [c for c in joined.columns if c.endswith('#')]
pct_columns = [c for c in joined.columns if c.startswith('Percent')]
pct_columns += [c for c in joined.columns if c.startswith('%')]
pct_columns += [c for c in joined.columns if c.endswith('%')]

joined.sort_values('Difference Ratio').head(3).style. \
    format('{:.2f}', subset=f2_columns). \
    format('{:.1%}', subset=pct_columns)
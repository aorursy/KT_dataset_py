import pandas as pd
import numpy as np
%%time

# 2016 Season Data Processing
ngs_2016_pre = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-pre.csv')
ngs_2016_1_6 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk1-6.csv')
ngs_2016_7_12 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk7-12.csv')
ngs_2016_13_17 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-reg-wk13-17.csv')
ngs_2016_post = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2016-post.csv')

# Combine
ngs_2016 = pd.concat([ngs_2016_pre, ngs_2016_1_6, ngs_2016_7_12, ngs_2016_13_17, ngs_2016_post], axis=0)

# Clear up memory
del ngs_2016_pre
del ngs_2016_1_6
del ngs_2016_7_12
del ngs_2016_13_17
del ngs_2016_post

# 2017 Season Data Processing
ngs_2017_pre = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-pre.csv')
ngs_2017_1_6 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk1-6.csv')
ngs_2017_7_12 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk7-12.csv')
ngs_2017_13_17 = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-reg-wk13-17.csv')
ngs_2017_post = pd.read_csv('../input/NFL-Punt-Analytics-Competition/NGS-2017-post.csv')

# Combine
ngs_2017 = pd.concat([ngs_2017_pre, ngs_2017_1_6, ngs_2017_7_12, ngs_2017_13_17, ngs_2017_post], axis=0)

# Clear up memory
del ngs_2017_pre
del ngs_2017_1_6
del ngs_2017_7_12
del ngs_2017_13_17
del ngs_2017_post

# Combine
ngs_all = pd.concat([ngs_2016, ngs_2017], axis=0)

# Clear up memory
del ngs_2016
del ngs_2017

# Drop unneeded columns
droppers = ['Season_Year', 'o', 'dir']
ngs_all.drop(columns=droppers, inplace=True)
# Fair Catch
fair_catch_df = pd.read_csv('../input/ngsconcussion/play-fair_catch.csv')
remainder_df = fair_catch_df.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})

# Create condensed set of NGS data
condensed_ngs = pd.merge(remainder_df, ngs_all,
                          how='inner',
                          on=['GameKey', 'PlayID'])

condensed_ngs.to_csv('NGS-fair_catch.csv', index=False)
# Punt Return
fair_catch_df = pd.read_csv('../input/ngsconcussion/play-punt_return.csv')
remainder_df = fair_catch_df.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})

# Create condensed set of NGS data
condensed_ngs = pd.merge(remainder_df, ngs_all,
                          how='inner',
                          on=['GameKey', 'PlayID'])

condensed_ngs.to_csv('NGS-punt_return.csv', index=False)
# Concussion
concussion_df = pd.read_csv('../input/NFL-Punt-Analytics-Competition/video_review.csv')
remainder_df = concussion_df.groupby(['GameKey','PlayID']).size().reset_index().rename(columns={0:'count'})

# Create condensed set of NGS data
condensed_ngs = pd.merge(remainder_df, ngs_all,
                          how='inner',
                          on=['GameKey', 'PlayID'])

condensed_ngs.to_csv('NGS-concussion.csv', index=False)

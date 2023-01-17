import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



dataframe = pd.read_csv('../input/nfl_draft.csv')

dataframe.head()
av_df = dataframe[dataframe.Year <= 2011]

av_df.head()
print("total rows: " + str(len(av_df)))

print("Average First4AV: " + str(np.mean(av_df.First4AV)))

print("Median First4AV: " + str(np.median(av_df.First4AV)))
ts = av_df[["Year", "Rnd", "Pick", "First4AV"]]

ts1 = ts.groupby(['Pick']).agg({'First4AV' : [np.mean, np.median]})

ts1
rolledDF = pd.DataFrame(ts1.to_records()) #flatten our multi-index table

rolledDF.columns = ['Pick', 'F4AV_mean', 'F4AV_median']

rolledDF['F4AV_Relative_to_AV'] = rolledDF['F4AV_mean'] - rolledDF['F4AV_median']

rolledDF
first_round_df = av_df[av_df['Rnd'] == 1.0]

second_round_df = av_df[av_df['Rnd'] == 2.0]

third_round_df = av_df[av_df['Rnd'] == 3.0]

the_field_df = av_df



first_round_df.head()
# df[['Year', 'YearsPlayed']].boxplot(by='Year')

fg = first_round_df[['Pick', 'First4AV']].boxplot(by = 'Pick')

fg.set_xlabel('Pick')

fg.set_ylabel('F4AV')

fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 12

fig_size[1] = 6

plt.rcParams["figure.figsize"] = fig_size

fg
fg = first_round_df[['Pick','First4AV']].boxplot(by = 'Pick')
fg = second_round_df[['Pick', 'First4AV']].boxplot(by = 'Pick')
fg = third_round_df[['Pick', 'First4AV']].boxplot(by = 'Pick')
# create a new temp set pivoting on position this time

ts = av_df[["Year", "Rnd", "Pick", "Position Standard", "First4AV"]]

ts2 = ts.groupby(['Position Standard']).agg({'First4AV' : [np.mean, np.median, np.max]})



#flatten the multi-index table and sort on mean descending

rolled_ts2 = pd.DataFrame(ts2.to_records()) #flatten our multi-index table

rolled_ts2.columns = ['Position', 'F4AV_mean', 'F4AV_median', 'F4AV_max']

rolled_ts2['F4AV_Relative_to_AV'] = rolled_ts2['F4AV_mean'] - rolled_ts2['F4AV_median']

# rolled_ts2.sort(['F4AV_mean'], ascending = [0])

rolled_ts2.sort_values(by='F4AV_mean', ascending = False)
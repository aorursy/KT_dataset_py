import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
superbowl_df = pd.read_csv('/kaggle/input/superbowl-history-1967-2020/superbowl.csv')    #Reading superbowl dataset

superbowl_df.head(5)    #Show first five elements
print('DataFrame Size : ', superbowl_df.shape)    #checking dataset size

superbowl_df.info()  #getting informations about the dataset
superbowl_df.drop(['SB'], axis=1, inplace=True)    #dropping column 'SB'

superbowl_df.head(5)
superbowl_groupby_winner = superbowl_df.groupby(by='Winner').count()    ##grouping data by 'Winner' coulmn value

superbowl_groupby_winner
superbowl_groupby_winner.reset_index(inplace=True)    #reset index

superbowl_groupby_winner.rename(columns = {"Date": "Count"}, inplace=True)    ##remand 'Date' coulmn

superbowl_groupby_winner
plot = sns.barplot(x = 'Count', y = 'Winner', data = superbowl_groupby_winner, orient = "h").set_title('Superbowl winning teams!')
superbowl_groupby_mvp = superbowl_df.groupby(by='MVP').count()

superbowl_groupby_mvp.reset_index(inplace=True)

superbowl_groupby_mvp.rename(columns = {"Date": "Count"}, inplace=True)
plot = sns.barplot(x = 'Count', y = 'MVP', data = superbowl_groupby_mvp, orient = "h").set_title('Superbowl MVP!')
plt.figure(figsize=(20,10))    #modify figure size

plot = sns.barplot(x = 'Count', y = 'MVP', data = superbowl_groupby_mvp, orient = "h").set_title('Superbowl MVP!')
superbowl_df['point difference'] = superbowl_df['Winner Pts'] - superbowl_df['Loser Pts']    #create new column

superbowl_df.head(3)
superbowl_df['point difference'].describe()    #get stats for numerical values
superbowl_df.boxplot(column = 'point difference')
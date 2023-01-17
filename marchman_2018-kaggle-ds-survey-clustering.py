import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Opening the file.
multipleChoice_df = pd.read_csv('../input/multipleChoiceResponses.csv')
#Create a list of all columns. Remove columns that you don't want to delete. Drop only the columns in the list.
target_df = multipleChoice_df
target_col_list = list(target_df.columns.values)
target_col_list.remove('Q6')
target_col_list.remove('Q40')
for col in target_col_list:
    target_df.drop([col], axis=1,inplace=True)
target_df.head()
for column in target_df.columns:
    print('Null values in',str(column),'?',target_df[column].isnull().any())
clean_target_df = target_df.dropna()
clean_target_df.isnull().any()
group_target_df = clean_target_df.groupby(['Q6','Q40']).size().reset_index(name='Count')
group_target_df.head(10)
pivot_target_df = group_target_df.pivot(index='Q6',columns='Q40',values='Count')
pivot_target_df
pivot_target_df.drop(['Which better demonstrates expertise in data science: \
academic achievements or independent projects? - Your views:'], axis=1,inplace=True)
pivot_target_df.drop(pivot_target_df.index[18],inplace=True)
give_me_cluster_df = pivot_target_df.fillna(0)
give_me_cluster_df
#sns.clustermap creates the desired plot. fmt='g' adjusts for scientific notation of digits in cells.
#The second line calls the .ax_heatmap to give a slight rotation to the x axis labels.
graph = sns.clustermap(give_me_cluster_df,cmap='magma', annot=True, fmt='g')
plt.setp(graph.ax_heatmap.xaxis.get_majorticklabels(), rotation=85)
plt.show()
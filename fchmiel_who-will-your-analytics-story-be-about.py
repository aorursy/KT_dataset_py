import numpy as np

import pandas as pd

pd.set_option("max_columns", 200)        

from umap import UMAP

import matplotlib.pyplot as plt

import plotly.express as px

import warnings

warnings.filterwarnings('ignore')

from wordcloud import WordCloud

from sklearn.cluster import KMeans

from matplotlib_venn import venn3
# load the survey results

df = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')



# drop the first row containing the questions



df.drop(0, inplace=True)
# create boolean masks of people who identify as Female and who are located in India

gender_mask = df['Q2']=='Female'

country_mask = df['Q3']=='India'



df_subset = df[gender_mask & country_mask]



# you can then do your analysis, lets look at the mode of each column:

df_subset.mode().iloc[0:1,] # iloc is a hack to drop some NaN column
jobcloud = WordCloud(background_color='white').generate(" ".join(df_subset['Q5'].dropna()))



fig = plt.figure()

plt.imshow(jobcloud)

plt.axis('off')

fig.set_size_inches(10,7)
# Make a list of all countries in SA

south_american_countries = ['Brazil', 'Colombia', 'Argentina','Peru','Venezuela',

                            'Chile', 'Ecuador', 'Bolivia', 'Paraguay', 'Uruguay',

                            'Guyana', 'Suriname', 'French Guiana']



# create boolean mask and use it

SA_mask = df['Q3'].isin(south_american_countries)

df_subset = df[SA_mask]



# again look at the mode of our subset:

df_subset.mode().iloc[0:1,] # iloc is a hack to drop some NaN columns
# count number of respondents in each age group

age_group_cnt = df_subset.groupby('Q1')['Q2'].count()



# plot the graph

fig, ax = plt.subplots()

ax.bar(age_group_cnt.index, age_group_cnt.values, color='gray', alpha=0.80)



# add axis labels etc.

ax.set_xlabel('Age Group')

ax.set_ylabel('Number of respondents')

ax.set_title('Age of of respondents in South America')

_ = plt.setp(ax.get_xticklabels(), rotation=90)

fig.set_size_inches(6,4)
salary_mapping = {'$0-999':'low', '1,000-1,999':'low', 

                  '10,000-14,999':'low', '100,000-124,999':'high',

                  '125,000-149,999':'high', '15,000-19,999':'low', 

                  '150,000-199,999':'high', '2,000-2,999':'low',

                  '20,000-24,999':'low', '200,000-249,999':'high', 

                  '25,000-29,999':'medium', '250,000-299,999':'high',

                  '3,000-3,999':'low','30,000-39,999':'medium',

                  '300,000-500,000':'high', '4,000-4,999':'low',

                  '40,000-49,999':'medium', '5,000-7,499':'low', 

                  '50,000-59,999':'medium', '60,000-69,999':'medium',

                  '7,500-9,999':'low', '70,000-79,999':'medium', 

                  '80,000-89,999':'medium', '90,000-99,999':'medium',

                  '> $500,000':'high'}



# create new column for the income group and convert the old salary

df['income_group'] = df['Q10'].map(salary_mapping)



# check the number of respondents in each income group

df.groupby('income_group')['Q1'].count()
# Using our new group, we can then use boolean masking (example 1) to look at a particular earning bracket:



income_mask = df['income_group'] == 'high'



df_subset = df[income_mask]

df_subset.mode().iloc[0:1,]

df.loc[~df['Q2'].isin(['Male', 'Female']), 'Q2'] = 'Other'



fig, axes = plt.subplots(nrows=1, ncols=3)



for ax, income in zip(axes, ['low','medium','high']):

    df_income = df[df['income_group']==income]

    gender_count = df_income.groupby('Q2')['Q1'].count()

    ax.pie(gender_count.values, labels=gender_count.index, autopct='%.1f',

           colors=['#a6d99c','#b19cd9','#d9d09c'])

    ax.set_title(income.capitalize() + ' income')



fig.set_size_inches(12,5)
# get a df with just media questions in it

columns_to_cluster = df.columns.str.contains('Q12')

df_media = df.loc[:, columns_to_cluster]



# convert it to a binary df

df_media = pd.get_dummies(df_media).iloc[:,0:10]



# clean up the column names

new_col_names = [col.split('_')[-1].split('(')[0].strip() for col in df_media.columns]

df_media.columns = new_col_names



# optionally drop anyone who didn't select any media interaction

drop_mask = ~(df_media.sum(axis=1)==0)

df_media = df_media[drop_mask]

df_subset = df[drop_mask]
# perform the clustering

y_pred = KMeans(n_clusters=3, random_state=42, max_iter=10000).fit_predict(df_media.values)



# add the cluster identification to the df



df_media['cluster_number'] = y_pred
# inspect the different clustering

cluster_sizes = df_media.groupby('cluster_number').sum()

cluster_sizes
fig, axes = plt.subplots(1, 3)



for i, (ax, cluster) in enumerate(zip(axes, cluster_sizes.index)):

    # get the top three media types used in the cluster

    values = cluster_sizes.loc[cluster,:]

    top_three = values.sort_values()[-3:]

    top_3_names = list(top_three.index)

    # create the venn diagram, 

    masks = [(df_media[top_3_names[i]]==0, df_media[top_3_names[i]]==1) for i in [0,1,2]]

    venn3(subsets=(len(df_media.loc[masks[0][1] & masks[1][0] & masks[2][0]]),

                   len(df_media.loc[masks[0][0] & masks[1][1] & masks[2][0]]),

                   len(df_media.loc[masks[0][1] & masks[1][1] & masks[2][0]]),

                   len(df_media.loc[masks[0][0] & masks[1][0] & masks[2][1]]),

                   len(df_media.loc[masks[0][1] & masks[1][0] & masks[2][1]]),

                   len(df_media.loc[masks[0][0] & masks[1][1] & masks[2][1]]),

                   len(df_media.loc[masks[0][1] & masks[1][1] & masks[2][1]])),

          set_labels=(top_3_names[0], top_3_names[1], top_3_names[2]), 

          ax=ax)

    # add titles to the plots

    ax.set_title(f'Cluster {i+1}', fontsize=14)



fig.set_size_inches(20,8)
# assign clusters to original df

df_subset['clusters'] = y_pred+1
# look at mode of cluster 2

df_subset[df_subset['clusters']==2].mode()
# look at mode of cluster 3

df_subset[df_subset['clusters']==3].mode()
# select just a subsection of the questions to cluster on

questions_to_use = ['Q4','Q5','Q6','Q14']

# drop the first row

try:

    df.drop(0, inplace=True, axis=0)

except KeyError:

    print('Row 0 does not exist')

column_mask =1
# one hot encode the questions

encoded_df = pd.get_dummies(df[questions_to_use])



# make the column names more readable

stripped_columns = [col.split('_')[1] if not col.endswith('Other') else col for col in encoded_df.columns]

encoded_df.columns = stripped_columns
umap_params = {'metric':'hamming', # hamming is a boolean distance metric

               'n_neighbors':500, # focus more on global structure

              'random_state':1, # use the random seeds to keep output reproducible

               'transform_seed':1

              }
embedder = UMAP(**umap_params)



X_embedded = embedder.fit_transform(encoded_df)



# add the coords in the 2D embedded space  for each instance

encoded_df['x'] = X_embedded[:,0]

encoded_df['y'] = X_embedded[:,1]



# add income

encoded_df['income_group'] = df['income_group']

fig = px.scatter(encoded_df, x="x", y="y", hover_data=encoded_df.columns[:-2], color='Data Scientist')

fig.show()
fig = px.scatter(encoded_df, x="x", y="y", hover_data=encoded_df.columns[:-2], color='Data Scientist')

fig.layout.xaxis.range = (16.5,19)

fig.layout.yaxis.range = (-50,-47)

fig.show()
encoded_df.loc[encoded_df['income_group'].isna(), 'income_group'] = 'No information'

fig = px.scatter(encoded_df, x="x", y="y", hover_data=encoded_df.columns[:-2], color='income_group', opacity=0.5)

fig.show()
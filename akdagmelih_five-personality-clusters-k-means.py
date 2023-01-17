import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
data_raw = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')

data = data_raw.copy()

pd.options.display.max_columns = 150



data.drop(data.columns[50:107], axis=1, inplace=True)

data.drop(data.columns[51:], axis=1, inplace=True)



print('Number of participants: ', len(data))

data.head()
print('Is there any missing value? ', data.isnull().values.any())

print('How many missing values? ', data.isnull().values.sum())

data.dropna(inplace=True)

print('Number of participants after eliminating missing values: ', len(data))
# Participants' nationality distriution

countries = pd.DataFrame(data['country'].value_counts())

countries_5000 = countries[countries['country'] >= 5000]

plt.figure(figsize=(15,5))

sns.barplot(data=countries_5000, x=countries_5000.index, y='country')

plt.title('Countries With More Than 5000 Participants')

plt.ylabel('Participants');
# Groups and Questions

ext_questions = {'EXT1' : 'I am the life of the party',

                 'EXT2' : 'I dont talk a lot',

                 'EXT3' : 'I feel comfortable around people',

                 'EXT4' : 'I keep in the background',

                 'EXT5' : 'I start conversations',

                 'EXT6' : 'I have little to say',

                 'EXT7' : 'I talk to a lot of different people at parties',

                 'EXT8' : 'I dont like to draw attention to myself',

                 'EXT9' : 'I dont mind being the center of attention',

                 'EXT10': 'I am quiet around strangers'}



est_questions = {'EST1' : 'I get stressed out easily',

                 'EST2' : 'I am relaxed most of the time',

                 'EST3' : 'I worry about things',

                 'EST4' : 'I seldom feel blue',

                 'EST5' : 'I am easily disturbed',

                 'EST6' : 'I get upset easily',

                 'EST7' : 'I change my mood a lot',

                 'EST8' : 'I have frequent mood swings',

                 'EST9' : 'I get irritated easily',

                 'EST10': 'I often feel blue'}



agr_questions = {'AGR1' : 'I feel little concern for others',

                 'AGR2' : 'I am interested in people',

                 'AGR3' : 'I insult people',

                 'AGR4' : 'I sympathize with others feelings',

                 'AGR5' : 'I am not interested in other peoples problems',

                 'AGR6' : 'I have a soft heart',

                 'AGR7' : 'I am not really interested in others',

                 'AGR8' : 'I take time out for others',

                 'AGR9' : 'I feel others emotions',

                 'AGR10': 'I make people feel at ease'}



csn_questions = {'CSN1' : 'I am always prepared',

                 'CSN2' : 'I leave my belongings around',

                 'CSN3' : 'I pay attention to details',

                 'CSN4' : 'I make a mess of things',

                 'CSN5' : 'I get chores done right away',

                 'CSN6' : 'I often forget to put things back in their proper place',

                 'CSN7' : 'I like order',

                 'CSN8' : 'I shirk my duties',

                 'CSN9' : 'I follow a schedule',

                 'CSN10' : 'I am exacting in my work'}



opn_questions = {'OPN1' : 'I have a rich vocabulary',

                 'OPN2' : 'I have difficulty understanding abstract ideas',

                 'OPN3' : 'I have a vivid imagination',

                 'OPN4' : 'I am not interested in abstract ideas',

                 'OPN5' : 'I have excellent ideas',

                 'OPN6' : 'I do not have a good imagination',

                 'OPN7' : 'I am quick to understand things',

                 'OPN8' : 'I use difficult words',

                 'OPN9' : 'I spend time reflecting on things',

                 'OPN10': 'I am full of ideas'}



# Group Names and Columns

EXT = [column for column in data if column.startswith('EXT')]

EST = [column for column in data if column.startswith('EST')]

AGR = [column for column in data if column.startswith('AGR')]

CSN = [column for column in data if column.startswith('CSN')]

OPN = [column for column in data if column.startswith('OPN')]
# Defining a function to visualize the questions and answers distribution

def vis_questions(groupname, questions, color):

    plt.figure(figsize=(40,60))

    for i in range(1, 11):

        plt.subplot(10,5,i)

        plt.hist(data[groupname[i-1]], bins=14, color= color, alpha=.5)

        plt.title(questions[groupname[i-1]], fontsize=18)
print('Q&As Related to Extroversion Personality')

vis_questions(EXT, ext_questions, 'orange')
print('Q&As Related to Neuroticism Personality')

vis_questions(EST, est_questions, 'pink')
print('Q&As Related to Agreeable Personality')

vis_questions(AGR, agr_questions, 'red')
print('Q&As Related to Conscientious Personality')

vis_questions(CSN, csn_questions, 'purple')
print('Q&As Related to Open Personality')

vis_questions(OPN, opn_questions, 'blue')
# For ease of calculation lets scale all the values between 0-1 and take a sample of 5000 

from sklearn.preprocessing import MinMaxScaler



df = data.drop('country', axis=1)

columns = list(df.columns)



scaler = MinMaxScaler(feature_range=(0,1))

df = scaler.fit_transform(df)

df = pd.DataFrame(df, columns=columns)

df_sample = df[:5000]
# Visualize the elbow

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer



kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,15))

visualizer.fit(df_sample)

visualizer.poof()
# Creating K-means Cluster Model

from sklearn.cluster import KMeans



# I use the unscaled data but without the country column

df_model = data.drop('country', axis=1)



# I define 5 clusters and fit my model

kmeans = KMeans(n_clusters=5)

k_fit = kmeans.fit(df_model)
# Predicting the Clusters

pd.options.display.max_columns = 10

predictions = k_fit.labels_

df_model['Clusters'] = predictions

df_model.head()
df_model.Clusters.value_counts()
pd.options.display.max_columns = 150

df_model.groupby('Clusters').mean()
# Summing up the different questions groups

col_list = list(df_model)

ext = col_list[0:10]

est = col_list[10:20]

agr = col_list[20:30]

csn = col_list[30:40]

opn = col_list[40:50]



data_sums = pd.DataFrame()

data_sums['extroversion'] = df_model[ext].sum(axis=1)/10

data_sums['neurotic'] = df_model[est].sum(axis=1)/10

data_sums['agreeable'] = df_model[agr].sum(axis=1)/10

data_sums['conscientious'] = df_model[csn].sum(axis=1)/10

data_sums['open'] = df_model[opn].sum(axis=1)/10

data_sums['clusters'] = predictions

data_sums.groupby('clusters').mean()
# Visualizing the means for each cluster

dataclusters = data_sums.groupby('clusters').mean()

plt.figure(figsize=(22,3))

for i in range(0, 5):

    plt.subplot(1,5,i+1)

    plt.bar(dataclusters.columns, dataclusters.iloc[:, i], color='green', alpha=0.2)

    plt.plot(dataclusters.columns, dataclusters.iloc[:, i], color='red')

    plt.title('Cluster ' + str(i))

    plt.xticks(rotation=45)

    plt.ylim(0,4);
# In order to visualize in 2D graph I will use PCA

from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_fit = pca.fit_transform(df_model)



df_pca = pd.DataFrame(data=pca_fit, columns=['PCA1', 'PCA2'])

df_pca['Clusters'] = predictions

df_pca.head()
plt.figure(figsize=(10,10))

sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='Clusters', palette='Set2', alpha=0.8)

plt.title('Personality Clusters after PCA');
my_data = pd.read_excel('../input/personalitytest/my_personality_test.xlsx')

my_data
my_personality = k_fit.predict(my_data)

print('My Personality Cluster: ', my_personality)
# Summing up the my question groups

col_list = list(my_data)

ext = col_list[0:10]

est = col_list[10:20]

agr = col_list[20:30]

csn = col_list[30:40]

opn = col_list[40:50]



my_sums = pd.DataFrame()

my_sums['extroversion'] = my_data[ext].sum(axis=1)/10

my_sums['neurotic'] = my_data[est].sum(axis=1)/10

my_sums['agreeable'] = my_data[agr].sum(axis=1)/10

my_sums['conscientious'] = my_data[csn].sum(axis=1)/10

my_sums['open'] = my_data[opn].sum(axis=1)/10

my_sums['cluster'] = my_personality

print('Sum of my question groups')

my_sums
my_sum = my_sums.drop('cluster', axis=1)

plt.bar(my_sum.columns, my_sum.iloc[0,:], color='green', alpha=0.2)

plt.plot(my_sum.columns, my_sum.iloc[0,:], color='red')

plt.title('Cluster 2')

plt.xticks(rotation=45)

plt.ylim(0,4);
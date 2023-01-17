# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
raw_data = pd.read_csv('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv',sep='\t')
pd.options.display.max_columns = 200

print('First 5 values of raw_data')

raw_data.head()
df = raw_data.copy()

df.drop(df.columns[108:], axis = 1, inplace = True)

df.dropna(inplace=True)

df.drop(df.columns[100:107],axis = 1, inplace = True)

df = df.loc[(df!=0).all(axis=1)] #remove entries with all 0 values on the questions

df.drop(df.columns[50:100],axis = 1, inplace = True)

df.drop(df[ df['country'] == 'NONE' ].index, inplace = True)

print('First 5 values in the data')

df.head()
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



neu_questions = {'EST1' : 'I get stressed out easily',

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

Extroversion = [column for column in df if (column.startswith('EXT') and not(column.__contains__('_E')))]

Neuroticism = [column for column in df if (column.startswith('EST') and not(column.__contains__('_E')))]

Agreeableness = [column for column in df if (column.startswith('AGR') and not(column.__contains__('_E')))]

Conscientiousness = [column for column in df if (column.startswith('CSN') and not(column.__contains__('_E')))]

Openness = [column for column in df if (column.startswith('OPN') and not(column.__contains__('_E')))]
# Defining a function to visualize the questions and answers distribution

def vis_questions(groupname, questions, color):

    plt.figure(figsize=(40,60))

    for i in range(1, 11):

        plt.subplot(10,5,i)

        plt.hist(df[groupname[i-1]], bins=10, color= color, alpha=0.6)

        plt.title(str(questions[groupname[i-1]]), fontsize=18)

        plt.xticks([1,2,3,4,5])

        plt.xlabel('Strongly Disagree   -   Disagree   -   Neutral   -   Agree   -   Strongly Agree')

        plt.ylabel('Number of Respondents')

        plt.subplots_adjust(hspace = 0.4)





print('Q&As Related to Extroversion')

vis_questions(Extroversion, ext_questions, 'blue')

print('Q&As Related to Neuroticism')

vis_questions(Neuroticism, neu_questions, 'green')
print('Q&As Related to Agreeableness')

vis_questions(Agreeableness, agr_questions, 'red')

print('Q&As Related to Conscientiousness')

vis_questions(Conscientiousness, csn_questions, 'orange')
print('Q&As Related to Openness')

vis_questions(Openness, opn_questions, 'brown')
pd.options.display.max_rows = 250

print('Number of countries of the participants: ',len(df.country.value_counts()))

print(df.country.value_counts().head(20))

print('...')

change_scale =['EXT2','EXT4','EXT6','EXT10','EXT8','EST2','EST4','AGR1','AGR3','AGR5','AGR7','CSN2','CSN4','CSN6','CSN8','OPN2',\

               'OPN4','OPN6'] 

#we change the scale for those columns because for those questions, high score actually means the individual/participant is low on the trait that 

#the question is related to or vice versa, for example, for the question EST2 'I am relaxed most of the time', individuals who score high (4 or 5)

#on the question is actually showing traits of low neuroticsm



df_excl = df.groupby('country', as_index = False, group_keys = False).filter(lambda x: len(x) >= 1000) 

#remove countries where value count is less than 1000

sample = df_excl.groupby('country',as_index = False,group_keys=False).apply(lambda s: s.sample(1000,replace = True, random_state = 1))

#sample 1000 values from each country



sample[change_scale] = 6 - sample[change_scale]

sample.head()
sample_averaged = sample.groupby('country',as_index = True,group_keys=False).mean()

sample_stddev = sample.groupby('country',as_index = True,group_keys=False).std()



questions = {'EXT1' : 'I am the life of the party',

                 'EXT2' : 'I dont talk a lot',

                 'EXT3' : 'I feel comfortable around people',

                 'EXT4' : 'I keep in the background',

                 'EXT5' : 'I start conversations',

                 'EXT6' : 'I have little to say',

                 'EXT7' : 'I talk to a lot of different people at parties',

                 'EXT8' : 'I dont like to draw attention to myself',

                 'EXT9' : 'I dont mind being the center of attention',

                 'EXT10': 'I am quiet around strangers',

                 'EST1' : 'I get stressed out easily',

                 'EST2' : 'I am relaxed most of the time',

                 'EST3' : 'I worry about things',

                 'EST4' : 'I seldom feel blue',

                 'EST5' : 'I am easily disturbed',

                 'EST6' : 'I get upset easily',

                 'EST7' : 'I change my mood a lot',

                 'EST8' : 'I have frequent mood swings',

                 'EST9' : 'I get irritated easily',

                 'EST10': 'I often feel blue',

                 'AGR1' : 'I feel little concern for others',

                 'AGR2' : 'I am interested in people',

                 'AGR3' : 'I insult people',

                 'AGR4' : 'I sympathize with others feelings',

                 'AGR5' : 'I am not interested in other peoples problems',

                 'AGR6' : 'I have a soft heart',

                 'AGR7' : 'I am not really interested in others',

                 'AGR8' : 'I take time out for others',

                 'AGR9' : 'I feel others emotions',

                 'AGR10': 'I make people feel at ease',

                 'CSN1' : 'I am always prepared',

                 'CSN2' : 'I leave my belongings around',

                 'CSN3' : 'I pay attention to details',

                 'CSN4' : 'I make a mess of things',

                 'CSN5' : 'I get chores done right away',

                 'CSN6' : 'I often forget to put things back in their proper place',

                 'CSN7' : 'I like order',

                 'CSN8' : 'I shirk my duties',

                 'CSN9' : 'I follow a schedule',

                 'CSN10' : 'I am exacting in my work',

                 'OPN1' : 'I have a rich vocabulary',

                 'OPN2' : 'I have difficulty understanding abstract ideas',

                 'OPN3' : 'I have a vivid imagination',

                 'OPN4' : 'I am not interested in abstract ideas',

                 'OPN5' : 'I have excellent ideas',

                 'OPN6' : 'I do not have a good imagination',

                 'OPN7' : 'I am quick to understand things',

                 'OPN8' : 'I use difficult words',

                 'OPN9' : 'I spend time reflecting on things',

                 'OPN10': 'I am full of ideas'}



colors = ['blue', 'green', 'red', 'orange', 'brown']



plt.figure(figsize=(20,200))

for i in range(1,len(sample_averaged.columns)):

    plt.subplot(50,1,i)

    plt.plot(sample_averaged.index, sample_averaged[sample_averaged.columns[i]], color = colors[(i-1)//10],alpha = .7,\

             marker='s',linewidth=3, markersize=6)

    plt.errorbar(sample_averaged.index, sample_averaged[sample_averaged.columns[i]],sample_stddev[sample_stddev.columns[i]])

    plt.yticks([1,2,3,4,5])

    plt.xlabel('Country Code')

    plt.title(questions[sample_averaged.columns[i]], fontsize=18);

    plt.grid(axis='y',b=True, which='major', color='#666666', linestyle='-')

    plt.subplots_adjust(hspace = 0.4)

      

plt.savefig('QuestionsbyCountries.pdf')
 #we want to see the correlation between questions

questions_corr = pd.DataFrame.corr(sample)

print('Questions Mapping')

print('1 to 10 is regarding Extroversion')

print('11 to 20 is regarding Neuroticism')

print('21 to 30 is regarding Agreeableness')

print('31 to 40 is regarding Conscientiousness')

print('41 to 50 is regarding Openness')



    

plt.figure(figsize = (10,8.5))

plt.pcolor(questions_corr, cmap='plasma');

plt.grid(b=True, which='major', color='k', linestyle='-');

plt.colorbar();

plt.title('Questions Correlation Matrix', fontsize = 18)

plt.xlabel('Extroversion   -   Neuroticism   -   Agreeableness   -   Conscientiousness   -   Openness')

plt.ylabel('Extroversion   -   Neuroticism   -   Agreeableness   -   Conscientiousness   -   Openness')

plt.savefig('Question_correlation.pdf')
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

scaler = StandardScaler()

data = sample.drop('country', axis = 1)

data_scaled = scaler.fit_transform(data)



# fitting multiple k-means algorithms and storing the values in an empty list

SSE = []

for cluster in range(1,20):

    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')

    kmeans.fit(data_scaled)

    SSE.append(kmeans.inertia_)



# converting the results into a dataframe and plotting them

frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})

plt.figure(figsize=(12,6))

plt.plot(frame['Cluster'], frame['SSE'], marker='o')

plt.xlabel('Number of clusters')

plt.ylabel('Loss Function (Inertia)')
# k means using 9 clusters and k-means++ initialization

kmeans = KMeans(n_jobs = -1, n_clusters = 9, init='k-means++')

kmeans.fit(data_scaled)

pred = kmeans.predict(data_scaled)



data['Cluster'] = pred

data_clustered = data.groupby('Cluster').mean()



print('X-axis Mapping')

print('1 to 10 is the region of Extroversion')

print('11 to 20 is the region of Neuroticism')

print('21 to 30 is the region of Agreeableness')

print('31 to 40 is the region of Conscientiousness')

print('41 to 50 is the region of Openness')



plt.figure(figsize=(20,20))

for i in data_clustered.index:

    plt.subplot(len(data_clustered.index),1,i+1)

    plt.bar(range(1,51), data_clustered.iloc[i], color = colors[i%5],alpha = .6)

    plt.plot(range(1,51), data_clustered.iloc[i], color='black')

    plt.yticks([1,2,3,4,5])

    plt.title('Cluster ' + str(i+1), fontsize=18);

    plt.grid(b=True, which='major', color='#666666', linestyle='-')

    plt.subplots_adjust(hspace = 0.4)

    

plt.savefig('Clusters.pdf')
col_list = list(data_clustered.columns)

ext = col_list[0:10]

est = col_list[10:20]

agr = col_list[20:30]

csn = col_list[30:40]

opn = col_list[40:50]



data_sums = pd.DataFrame()

data_sums['extroversion'] = data_clustered[ext].sum(axis=1)

data_sums['neuroticism'] = data_clustered[est].sum(axis=1)

data_sums['agreeableness'] = data_clustered[agr].sum(axis=1)

data_sums['conscientiousness'] = data_clustered[csn].sum(axis=1)

data_sums['openness'] = data_clustered[opn].sum(axis=1)

data_sums



# Visualizing the means for each cluster

plt.figure(figsize=(23,3.5))

for i in range(0, len(data_sums.index)):

    plt.subplot(1, len(data_sums.index),i+1)

    plt.bar(data_sums.columns, data_sums.iloc[i], color='green', alpha=0.2)

    plt.plot(data_sums.columns, data_sums.iloc[i], color='red')

    plt.title('Cluster ' + str(i+1))

    plt.xticks(rotation=45)

    plt.ylim(0,50);

plt.tight_layout()

    

plt.savefig('Cluster_traits.pdf');
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
%matplotlib inline
import seaborn

import re

from collections import Counter

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv')

#Dropping Unnecessary Columns
df = df.drop(columns=['Unnamed: 0'], axis=1)
df.info()
df.head()
print('Feature\t\t\tNon-Null Count')
print('-------                 --------------')
for feature in df.columns: 
    print('{}\t{:>12}'.format(feature, df[feature][df[feature]!=-1].count()))
df['Company Name'].unique()
# Code from https://www.kaggle.com/nerdscoding/data-analyst-jobs-dataset-eda-with-plotly
df['Company Name'] = df['Company Name'].apply(lambda x: re.sub(r'\n.*','',str(x)))
df['Company Name'].unique()
df[df['Rating']==-1]
df['Rating'] = df['Rating'].replace({-1: 0})
for row in range(df.shape[0]): #df.shape[0]
#     print(df.loc[row, 'Salary Estimate'])
    lower, upper = df.loc[row, 'Salary Estimate'].split('-')
    try: 
        upper = int("".join(re.findall('\d+', upper))) * 1000
    except: 
        upper = 0
    try:
        lower = int("".join(re.findall('\d+', lower))) * 1000
    except: 
        lower = 0
#     print('{} {}'.format(upper, lower))
    df.loc[row, 'Upper Salary Estimate'] = upper
    df.loc[row, 'Lower Salary Estimate'] = lower
df['Location'].unique()
# Create new features of states and cities
tmp_split = df['Location'].str.split(',', expand=True)
df['City'] = tmp_split[0]
df['Country/ State'] = tmp_split[1].str.strip()
df['Size'].unique()
df['Size'] = df['Size'].replace({'-1': 'Unknown'})
df['Size'].unique()
plt.rcParams["figure.figsize"] = (20,10)
df['Company Name'].value_counts().sort_values(ascending=False).head(25).plot.bar()
plt.title('Companies with the Most Jobs Offered')
plt.show()
df['Company Name'].value_counts().sort_values(ascending=False).head(5)
# Junior data analyst
junior_jobs = df[df['Job Title'].str.contains(r'Data Analyst', na=True) & ~df['Job Title'].str.contains(r'Senior', na=True)]
senior_jobs = df[df['Job Title'].str.contains(r'Senior', na=True)]
junior_jobs.shape
plt.rcParams["figure.figsize"] = (10,5)
plt.title('Level of Data Analyst Jobs')
plt.bar(['Junior', 'Senior'], [junior_jobs.shape[0], senior_jobs.shape[0]], 0.5)
plt.show()
junior_jobs['Job Description']
junior_desc = []
for row in junior_jobs['Job Description'].to_list(): 
    row_content = row.split()
    row_content = [x.strip('.') for x in row_content]
    row_content = [x.strip(',') for x in row_content]
    for word in row_content: 
        junior_desc.append(word.lower())

junior_desc_counts = Counter(junior_desc) 
junior_desc_counts.most_common(20)
stopping_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'us', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'and/or',
                 'including', 'include', 'job', 'using', 'ability', 'across', 'related', 'provide', 'within', 'ensure', 'use', 'may', 'must', 'one', 'perform', 'also', 'meet', 'plus', 'impact', 'making', 'take', 'used']
stopping_words_with_first_cap = [stopping_word[0].upper() + stopping_word[1:] for stopping_word in stopping_words]

for stopping_word in stopping_words: 
    del junior_desc_counts[stopping_word]

for stopping_word in stopping_words_with_first_cap: 
    del junior_desc_counts[stopping_word]

punctuations = ['•', '-', '·', 'A', '&', '+', '/', '–']
for punctuation in punctuations: 
    del junior_desc_counts[punctuation]
junior_desc_counts.most_common(5)
plt.rcParams["figure.figsize"] = (25,10)
plt.title('Common Words in Junior Data Analyst Jobs')

selected = junior_desc_counts.most_common(20)
labels = [x[0] for x in selected]
values = [x[1] for x in selected]
plt.bar(labels, values)
plt.show()
senior_desc = []
for row in senior_jobs['Job Description'].to_list(): 
    row_content = row.split()
    row_content = [x.strip('.') for x in row_content]
    row_content = [x.strip(',') for x in row_content]
    for word in row_content: 
        senior_desc.append(word.lower())

senior_desc_counts = Counter(senior_desc) 

for stopping_word in stopping_words: 
    del senior_desc_counts[stopping_word]

for stopping_word in stopping_words_with_first_cap: 
    del senior_desc_counts[stopping_word]

for punctuation in punctuations: 
    del senior_desc_counts[punctuation]
senior_desc_counts.most_common(20)
plt.rcParams["figure.figsize"] = (25,10)
plt.title('Common Words in Senior Data Analyst Jobs')

selected = senior_desc_counts.most_common(20)
labels = [x[0] for x in selected]
values = [x[1] for x in selected]
plt.bar(labels, values)
plt.show()
skills_keyword = ['sql', 'excel', 'microsoft', 'python', 'analytics', 'programming', 'code', 'modeling', 'r', 'visualization'
                  , 'statistics', 'statistical'
                  , 'report', 'reporting', 'communication', 'documentation', 'tableau'
                 , 'strategies', 'management']
junior_skill_counts = []
senior_skill_counts = []

for data in skills_keyword: 
    junior_skill_counts.append(junior_desc_counts[data])
    senior_skill_counts.append(senior_desc_counts[data])


plt.rcParams["figure.figsize"] = (25,10)

bar_width = 0.35 #Width of the bar

junior_bar = plt.bar(np.arange(len(skills_keyword)), junior_skill_counts, bar_width, color='royalblue', label='Junior level')
senior_bar = plt.bar(np.arange(len(skills_keyword))+bar_width, senior_skill_counts, bar_width, color='seagreen', label='Senior level')
plt.xticks(np.arange(len(skills_keyword)), skills_keyword, rotation='vertical')

plt.title('Skills mentioned in Data Analysts Job Ads')
plt.ylabel('Frequency')
plt.legend()
plt.show()
plt.rcParams["figure.figsize"] = (8,6)
plt.boxplot([df['Lower Salary Estimate'],df['Upper Salary Estimate']], labels=['Min. Salary', 'Max. Salary'])

plt.title('Salary Distribution')
plt.show()
plt.rcParams["figure.figsize"] = (30,6)

# df.groupby('Industry')['Lower Salary Estimate']
df.boxplot(column='Lower Salary Estimate', by='Industry')

plt.title('Lower Salary Estimate by Industry')
plt.xticks(rotation='vertical')
plt.show()
plt.rcParams["figure.figsize"] = (25,6)

# df.groupby('Industry')['Lower Salary Estimate']
df.boxplot(column='Upper Salary Estimate', by='Industry')

plt.title('Upper Salary Estimate by Industry')
plt.xticks(rotation='vertical')
plt.show()
lowest_ind = df.groupby('Industry')['Lower Salary Estimate'].mean().idxmax()
lowest_ind_val = df.groupby('Industry')['Lower Salary Estimate'].mean().max()
print('The lowest expectedly paid industry is {} with ${}'.format(lowest_ind, lowest_ind_val))
highest_ind = df.groupby('Industry')['Upper Salary Estimate'].mean().idxmax()
highest_ind_val = df.groupby('Industry')['Upper Salary Estimate'].mean().max()
print('The highest expectedly paid industry is {} with ${}'.format(highest_ind, highest_ind_val))
plt.rcParams["figure.figsize"] = (25,6)

df.boxplot(column='Rating', by='Industry')

plt.title('Ratings by Industry')
plt.xticks(rotation='vertical')
plt.show()
print('Top 5 Average Rating by Industries')
df.groupby('Industry')['Rating'].mean().sort_values(ascending=False).head(5)
df.boxplot(column='Rating', by='Country/ State')

plt.title('Ratings by Industry')
plt.xticks(rotation='vertical')
plt.show()
plt.rcParams["figure.figsize"] = (25,6)

df.groupby('Country/ State')['Lower Salary Estimate'].mean().plot.bar()

plt.title('Min.Expected Salary by State')
plt.xlabel('State/ Country')
plt.ylabel('USD')
plt.show()
plt.rcParams["figure.figsize"] = (25,6)

df.groupby('Country/ State')['Upper Salary Estimate'].mean().plot.bar()

plt.title('Max.Expected Salary by State')
plt.xlabel('State/ Country')
plt.ylabel('USD')
plt.show()

import os

import re

import nltk

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.util import ngrams

from nltk.util import bigrams

from nltk.collocations import *

import pandas as pd

import numpy as np

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import seaborn
da_ny = pd.read_csv('../input/jobs_1.csv')
da_ny = pd.read_csv('../input/jobs_1.csv')

ba_ny = pd.read_csv('../input/jobs_2.csv')

ds_ny = pd.read_csv('../input/jobs_3.csv')

de_ny = pd.read_csv('../input/jobs_4.csv')

da_bo = pd.read_csv('../input/jobs_5.csv')

ba_bo = pd.read_csv('../input/jobs_6.csv')

ds_bo = pd.read_csv('../input/jobs_7.csv')

de_bo = pd.read_csv('../input/jobs_8.csv')

da_ch = pd.read_csv('../input/jobs_9.csv')

ba_ch = pd.read_csv('../input/jobs_10.csv')

ds_ch = pd.read_csv('../input/jobs_11.csv')

de_ch = pd.read_csv('../input/jobs_12.csv')

ds_main = pd.concat([da_ny,ba_ny,ds_ny,de_ny,da_bo,ba_bo,ds_bo,de_bo,da_ch,ba_ch,ds_ch,de_ch],ignore_index=True)

#ds_main = pd.concat([da_ny,da_bo,da_ch],ignore_index=True)
len(ds_main)
total_no_company=ds_main['company_name'].nunique()

print('Total number of firms with data science job vacancies',total_no_company)
ds_main.drop_duplicates(subset=['job_title','company_name','summary'],inplace = True)
len(ds_main)
ds_main.head()
ds_main_cities=ds_main.filter(['job_title','location'], axis=1)
ds_main_cities=ds_main_cities[ds_main_cities.location != 'NOT_FOUND']
len(ds_main_cities)

ds_main_cities.head()
ds_main_cities= ds_main_cities.groupby(['location'])['job_title'].count()
ds_main_cities.head()
ds_main_cities.to_csv('jobs_cities.csv',index=True)


most_vacancy= ds_main.groupby(['company_name'])['job_title'].count()

most_vacancy=most_vacancy.reset_index(name='position')

most_vacancy=most_vacancy.sort_values(['position'],ascending=False)

pareto_df=most_vacancy

most_vacancy.to_csv('jobs1.csv',index=True)

most_vacancy=most_vacancy.head(25)

print('Top 10 firms with most vacancies',most_vacancy)
fig, ax = plt.subplots(figsize = (10,6))

ax=seaborn.barplot(x="company_name", y="position", data=most_vacancy)    

ax.set_xticklabels(most_vacancy['company_name'],rotation=90)  

ax.set_xlabel('Companies',fontsize=16, color='black')

ax.set_ylabel('Number of Jobs',fontsize=16) 
# Finding total number of unique roles in data science domain from the given dataset

total_no_roles=ds_main['job_title'].nunique()

print('Total number of roles across all the firms',total_no_roles)



# most offered roles across all the firms

most_offd_roles=ds_main.groupby(['job_title'])['company_name'].count()   

most_offd_roles=most_offd_roles.reset_index(name='company_name')

most_offd_roles=most_offd_roles.sort_values(['company_name'],ascending=False)
most_offd_roles=most_offd_roles.head(30)   

print('Top 15 most wanted roles across firms',most_offd_roles)



most_offd_roles.to_csv('jobs2.csv',index=True)

# Plot graph for top most offered roles

fig,ax=plt.subplots(figsize=(12,6))

ax=seaborn.barplot(x="job_title", y="company_name", data=most_offd_roles)    

ax.set_xticklabels(most_offd_roles['job_title'],rotation=90)

ax.set_xlabel('MOST WANTED JOB ROLES',fontsize=10,color='black')

ax.set_ylabel('NO OF ROLES ACROSS INDUSTRY',fontsize=10,color='black')#
ds_main.head()

len(ds_main)

ds_main.summary[2]
ds_main.drop_duplicates(subset=['job_title','company_name','summary'],inplace = True)
len(ds_main)
summary=ds_main['summary']
summary.head(10)
summary = summary.apply(lambda x: " ".join(x.lower() for x in x.split()))

summary = summary.str.replace('[^\w\s]','')
stop = stopwords.words('english')

summary = summary.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

summary.head(10)
freq = pd.Series(' '.join(summary).split()).value_counts()[:10]

freq
freq = pd.Series(' '.join(summary).split()).value_counts()[-10:]

freq
freq = list(freq.index)

summary = summary.apply(lambda x: " ".join(x for x in x.split() if x not in freq))

summary.head()
#tokens = nltk.word_tokenize(summary)

type(summary)
docs=(summary.tolist())

summary_text = ''.join(docs)
tokens = nltk.word_tokenize(summary_text)
print(len(tokens))
tokens_pos_tag = nltk.pos_tag(tokens)
pos_df = pd.DataFrame(tokens_pos_tag, columns = ('word','POS'))
pos_df.head()
pos_sum = pos_df.groupby('POS', as_index=False).count() # group by POS tags
pos_sum.sort_values(['word'], ascending=[False]) # in descending order of number of words per tag
filtered_pos = [ ]
for one in tokens_pos_tag:

    if one[1] == 'NN' or one[1] == 'NNP' or one[1] == 'NNPS' or one[1] == 'JJ' or one[1] == 'NNS':

        filtered_pos.append(one)
print (len(filtered_pos))
fdist_pos = nltk.FreqDist(filtered_pos)
top_100_words = fdist_pos.most_common(100)
print(top_100_words)
top_words_df = pd.DataFrame(top_100_words, columns = ('pos','count'))
top_words_df.head()
top_words_df['Word'] = top_words_df['pos'].apply(lambda x: x[0]) # split the tuple of POS
top_words_df = top_words_df.drop('pos', 1) # drop the previous column
subset_pos = top_words_df[['Word', 'count']]



#tuples_pos = subset_pos.to_dict('split')

tuples_pos = [tuple(x) for x in subset_pos.values]

#tuples_pos=dict((x, y) for y, x in tuples_pos)

subset_pos.head()
d=dict(tuples_pos)

d['data'] = 4222

d.items()
from PIL import Image

wordcloud = WordCloud(colormap="Oranges")
wordcloud.generate_from_frequencies(d)
plt.figure()

plt.figure(figsize=(20,20))

plt.imshow(wordcloud, interpolation="bilinear")



plt.show()
bgs = nltk.bigrams(tokens)
fdist2 = nltk.FreqDist(bgs) # selecting bigrams from tokens
bgs_100 = fdist2.most_common(100) # top-100 bigrams

bgs_df = pd.DataFrame(bgs_100, columns = ('bigram','count'))

bgs_df.head()
bgs_df['phrase'] = bgs_df['bigram'].apply(lambda x: x[0]+" "+x[1]) # merging the tuple into a string
punctuation = re.compile(r'[-.?!,":;()|0-9]')

bgs_df['filter_bgs'] = bgs_df['phrase'].str.contains(punctuation) # finding strings with numbers and punctuation
bgs_df.head()
bgs_df = bgs_df[bgs_df.filter_bgs == False] # removing strings with numbers and punctuation
bgs_df = bgs_df.drop('bigram', 1)

bgs_df = bgs_df.drop('filter_bgs', 1) # removing the excess columns
bgs_df.reset_index()

bgs_df.head(10) #Final bigrams
tgs = nltk.ngrams(tokens,3)

fdist3 = nltk.FreqDist(tgs) # selecting trigrams from tokens

tgs_100 = fdist3.most_common(100) # top-100 trigrams

tgs_df = pd.DataFrame(tgs_100, columns = ('trigram','count'))

tgs_df.head()

tgs_df['phrase'] = tgs_df['trigram'].apply(lambda x: x[0]+" "+x[1]+" "+x[2])

                                           #" "+x[3]+" "+x[4])

                                           #+" "+x[3]) # merging the tuple into a string
tgs_df['filter_tgs'] = tgs_df['phrase'].str.contains(punctuation) # finding strings with numbers and punctuation

tgs_df.head()
tgs_df = tgs_df[tgs_df.filter_tgs == False] # removing strings with numbers and punctuation

tgs_df = tgs_df.drop('trigram', 1)

tgs_df = tgs_df.drop('filter_tgs', 1) # removing the excess columns
tgs_df.reset_index()

tgs_df.head(20) #Final trigrams
subset_pos1 = tgs_df[['phrase', 'count']]

tuples_pos1 = [tuple(x) for x in subset_pos1.values]

d4=dict(tuples_pos1)

#d4['data'] = 4269

d4.items()
wordcloud.generate_from_frequencies(d4)

plt.figure(figsize=(20,20))

plt.imshow(wordcloud, interpolation="bilinear", aspect='equal')

#plt.savefig('filename.jpg', dpi=200)

plt.show()

# Top data science lang (from KDnuggets) and popu langs from (TIOBE inex)



languages_list = ['R','Python','SAS','SQL','Scala','Java','Javascript','C#','.NET','VBA']

tools_list = ['Hadoop','MS Excel','Tableau','Qlik','Spark','Looker','Power BI','Alteryx']

libraries_list = ['ggplot','ggplot2','shiny','dplyr','tensorflow','scikit-learn','pandas','plotly']



skills_dict = {'languages':languages_list,'tools':tools_list,'libraries':libraries_list}

#^ = Not, \w alphaneumeric chars

#java or javascript

# ( space before or after R

# ds['R'] = ds['Job_Description'].str.contains(' R[^\w]', case=False).astype(int)

skills=[]

#should not be a part of a word [^\w]

for j in skills_dict.keys():

    

    for i in range(len(skills_dict[j])):

        i_regex = '[^\w]' + skills_dict[j][i] + '[^\w]'

        ds_main[skills_dict[j][i]] = ds_main['summary'].str.contains(i_regex, case=False).astype(int)

        total_count = sum(ds_main[skills_dict[j][i]])

        skills.append(skills_dict[j][i] + '  ' +  str(total_count))



type(skills)
str.split(skills[1])
pd.Series(skills)
sk=[]

for d in skills:

    sk2=str.split(d)

    sk.append(sk2)
sk2=pd.DataFrame.from_records(sk,columns=['lan','oc','x'])
sk2['lan'][13]='MS Excel'

sk2['oc'][13]=2

sk2['lan'][18]='Power BI'

sk2['oc'][18]=2
sk2=sk2.drop(['x'], axis=1)
sk2.oc = pd.to_numeric(sk2.oc, errors='coerce')
sk2=sk2.sort_values(by=['oc'],ascending=False)

sk2.head(20)
# Plot graph for top most offered roles

sk2.to_csv('jobs3.csv',index=True)

fig,ax=plt.subplots(figsize=(12,6))

ax=seaborn.barplot(x="lan", y="oc", data=sk2)    

ax.set_xticklabels(sk2['lan'],rotation=90)

ax.set_xlabel('Top tools in demand',fontsize=20,color='black')

ax.set_ylabel('Count',fontsize=16,color='black')#
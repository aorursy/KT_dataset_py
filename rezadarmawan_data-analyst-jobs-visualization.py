import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import re
data = pd.read_csv('../input/data-analyst-jobs/DataAnalyst.csv')
data.head()
data.columns
del data['Unnamed: 0']
data = data.replace([-1,'-1',-1.0], np.nan)

data.head()
for i in data.columns:

    print('Missing data in', i, data[i].isna().sum() / len(data)*100)
data = data.drop(columns=['Competitors','Easy Apply'])
data['Salary Estimate'] = data['Salary Estimate'].replace(np.nan,'0K-0K')

data.insert(len(data.columns),'Min Salary', np.nan)

data.insert(len(data.columns),'Max Salary', np.nan)

sal = [re.findall(r'\d+', i) for i in data['Salary Estimate']]

data['Max Salary'] = [int(i[1]) for i in sal]

data['Min Salary'] = [int(i[0]) for i in sal]



data.head()
ind_count = data['Industry'].value_counts().reset_index().sort_values(by=['Industry'])
fig1, ax1 = plt.subplots(figsize=(7,7))

ax1 = plt.barh(ind_count['index'][-20:], ind_count['Industry'][-20:])

plt.title('Industry with most job posting')
top_ind = ind_count['index'][-20:]

ind_rat = data.groupby("Industry")['Rating'].mean().reset_index()

top_ind_rat = ind_rat.loc[ind_rat['Industry'].isin(np.array(top_ind))].sort_values(by=['Rating'], ascending=False)

top_ind_rat.head()

fig = plt.figure(figsize=(15,7))

plt.bar(top_ind_rat['Industry'], top_ind_rat['Rating'], color ='maroon', width = 0.7) 

plt.xticks(rotation=45, ha='right')
import seaborn as sns



def comp_salary(comp, fixgap):

    fig2, axs = plt.subplots(1,2,figsize=(20,10))

    ax2 = sns.boxplot(x="Min Salary", y=comp, data=data, order=data[comp].value_counts()[:20].index, ax=axs[0])

    ax3 = sns.boxplot(x="Max Salary", y=comp, data=data, order=data[comp].value_counts()[:20].index, ax=axs[1])

    

    for i in range(len(axs)):

        axs[i].set_ylabel('')

        axs[i].tick_params(axis='x', which='major', labelsize=15)

        axs[i].xaxis.label.set_size(15)

        

    axs[0].set_yticks([])

    for label in axs[1].get_yticklabels():

        label.set_horizontalalignment('center')

        label.set_fontsize(16)

    axs[1].tick_params(axis='y', which='major', pad=(data[comp].value_counts()[:20].index.str.len().max()*4+fixgap), length=0)

    plt.close(2)

    plt.close(3)

    

    fig2.tight_layout()
comp_salary('Industry', 22)
size_rat = data.groupby("Size")['Rating'].mean().reset_index()

plt.bar(size_rat['Size'], size_rat['Rating'])

plt.xticks(rotation=45, ha='right')
loc_count = data['Location'].value_counts().reset_index().sort_values(by=['Location'], ascending=True)

fig2, ax2 = plt.subplots(figsize=(7,7))

ax2 = plt.barh(loc_count['index'][-20:], loc_count['Location'][-20:])

plt.title('City with Most Job Posting')
loc_rat = data.groupby("Location")['Rating'].mean().reset_index()

top_loc_rat = loc_rat.loc[loc_rat['Location'].isin(np.array(loc_count['index'][-20:]))].sort_values(by=['Rating'], ascending=False)



fig = plt.figure(figsize=(15,7))

plt.bar(top_loc_rat['Location'], top_loc_rat['Rating'], color ='maroon', width = 0.7) 

plt.xticks(rotation=45, ha='right')
comp_salary('Location', 15)
from collections import Counter

from string import punctuation

from wordcloud import WordCloud, STOPWORDS



# import nltk

# nltk.download('stopwords')

# nltk.download('punkt')



from nltk.corpus import stopwords

from nltk import word_tokenize



skill = ['sql', 'r', 'python', 'excel', 'visualization', 'statistic','html',

         'php','javascript','css','oracle', 'sas', 'etl','communication',

         'cloud','tableau', 'ssis', 'powerbi', 'google', 'statistical', 'presentation'

         'pandas', 'numpy', 'R', 'dashboard','report','ai','java','c++','linux'

         'finance','risk', 'financial','sales','product']



stoplist = set(stopwords.words('english') + list(punctuation))



texts = data['Job Description'].str.lower()



word_counts = Counter(word_tokenize('\n'.join(texts)))



skill_count = list(word_counts.items())

top_skill = [skill_count[i] for i in range(len(skill_count)) if(skill_count[i][0].lower() in skill)]



d = {}

for i in top_skill:

    d[i[0]] = i[1]
wordcloud = WordCloud(background_color = "white")

wordcloud.generate_from_frequencies(frequencies=d)

plt.figure(figsize=(7,7))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title('Most Common Knowledge')

plt.show()
texts_pos = data['Job Title'].str.lower()

word_counts_pos = Counter(word_tokenize('\n'.join(texts_pos)))

word_counts_pos.most_common(30)
text = data['Job Title'].values

wordcloud = WordCloud(background_color = 'white', stopwords = STOPWORDS).generate(str(text))

fig = plt.figure( figsize = (7, 7))

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
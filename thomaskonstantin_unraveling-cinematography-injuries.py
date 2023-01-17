# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import plotly.express as ex

import plotly.figure_factory as ff

import spacy



nlp = spacy.load('en')

inj_data = pd.read_csv('/kaggle/input/all-injuries-in-cinematography-19142019/movie_injury.csv')

inj_data.head(3)
inj_data['Caused Death'] = inj_data.Description.apply(lambda x: 1 if x.lower().find('death') != -1 else  0)
ex.pie(inj_data,'Caused Death',title='Proportion of injuries that lead to deaths',hole=0.3)
ex.histogram(inj_data,x='Year',marginal='rug',color='Caused Death')
from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud,STOPWORDS

import re 



raw_title_data = ' '

for word in inj_data.Title:

    tok = nlp(word.lower())

    tfs = [token.lemma_ for token in tok if token.is_stop == False]

    tfs = ' '.join(tfs)

    tfs = re.sub(r'[^a-z\s]', '', tfs) 

    raw_title_data += tfs+' '









wordcloud = WordCloud(width=800,height=800,background_color='white',min_font_size=5).generate(raw_title_data)

plt.figure(figsize=(20,11))

plt.imshow(wordcloud)

plt.axis('off')

plt.tight_layout(pad = 0) 

plt.show()
raw_desc_data = ' '

for word in inj_data.Description:

    tok = nlp(word.lower())

    tfs = [token.lemma_ for token in tok if token.is_stop == False]

    tfs = ' '.join(tfs)

    tfs = re.sub(r'[^a-z\s]', '', tfs) 

    raw_desc_data += tfs+' '

    

wordcloud = WordCloud(width=800,height=800,background_color='white',min_font_size=5).generate(raw_desc_data)

plt.figure(figsize=(20,11))

plt.imshow(wordcloud)

plt.axis('off')

plt.tight_layout(pad = 0) 

plt.show()
def clean_text_data(sir):

    tok = nlp(sir.lower())

    tfs = [token.lemma_ for token in tok if token.is_stop == False]

    tfs = ' '.join(tfs)

    tfs = re.sub(r'[^a-z\s]', '', tfs) 

    return tfs
inj_data.Description = inj_data.Description.apply(clean_text_data)

inj_data.Title = inj_data.Title.apply(clean_text_data)
vectorizer = TfidfVectorizer()

T_vectorizer = TfidfVectorizer()



s_matrix = vectorizer.fit_transform(inj_data.Description)

t_matrix = T_vectorizer.fit_transform(inj_data.Title)



desc_tfidf = pd.DataFrame(s_matrix.todense(),columns = vectorizer.get_feature_names())

title_tfidf = pd.DataFrame(t_matrix.todense(),columns = T_vectorizer.get_feature_names())
ex.imshow(desc_tfidf,title='Descreption TfIdf Heatmap')
ex.imshow(title_tfidf,title='Title TfIdf Heatmap')
from sklearn.decomposition import PCA



pca = PCA()



title_components = pca.fit_transform(title_tfidf)



exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)



ex.area(

    x=range(1, exp_var_cumul.shape[0] + 1),

    y=exp_var_cumul,

    title='Number of Components Needed To Explain Most Of Title TfIDF Feature Variance ',

    labels={"x": "# Components", "y": "Explained Variance"}

)
desc_pca = PCA()



desc_components = desc_pca.fit_transform(desc_tfidf)



exp_var_cumul = np.cumsum(desc_pca.explained_variance_ratio_)



ex.area(

    x=range(1, exp_var_cumul.shape[0] + 1),

    y=exp_var_cumul,

    title='Number of Components Needed To Explain Most Of Description TfIDF Feature Variance ',

    labels={"x": "# Components", "y": "Explained Variance"}

)
print('Desc tf-idf after pca shape: ',desc_components.shape)

print('Desc tf-idf before pca shape: ',desc_tfidf.shape)
desc_components_death_df = pd.DataFrame(desc_components)

desc_components_death_df = desc_components_death_df.add_prefix('PC_')

desc_components_death_df['Caused Death'] = inj_data['Caused Death']



ex.imshow(desc_components_death_df[desc_components_death_df['Caused Death']==1],title='Description PCs Heatmap Where An Injury Caused Death' )
title_components_death_df = pd.DataFrame(title_components)

title_components_death_df = title_components_death_df.add_prefix('PC_')

title_components_death_df['Caused Death'] = inj_data['Caused Death']



ex.imshow(title_components_death_df[title_components_death_df['Caused Death']==1],title='Title PCs Heatmap Where An Injury Caused Death' )
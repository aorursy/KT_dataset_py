import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from nltk.tokenize import sent_tokenize
import scipy.spatial

!pip install -U sentence-transformers

# Library from here: https://github.com/UKPLab/sentence-transformers
from sentence_transformers import SentenceTransformer
CLEAN_DATA_PATH = "../input/cord-19-eda-parse-json-and-generate-clean-csv/"

biorxiv_df = pd.read_csv(CLEAN_DATA_PATH + "biorxiv_clean.csv")
clean_pmc = pd.read_csv(CLEAN_DATA_PATH + "clean_pmc.csv")
papers_df = pd.concat([clean_pmc, biorxiv_df], axis=0).reset_index(drop=True)

papers_df.dropna(inplace=True)
papers_df.drop_duplicates(subset=['title'], keep=False, inplace=True)

# Load Sentence Embedding Model.
model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
papers_df.head()
sentences = sent_tokenize(papers_df.iloc[0]['text'])
sentences_df = pd.DataFrame({'id':np.zeros(len(sentences)).astype(int), 'sentences':sentences},index=None)

for i in range(1, len(papers_df)):
    paper_sentences = sent_tokenize(papers_df.iloc[i]['text'])
    paper_sentences_df = pd.DataFrame({'id':(np.ones(len(paper_sentences))*i).astype(int), 'sentences':paper_sentences},index=None)
    sentences_df = pd.concat([sentences_df, paper_sentences_df], axis=0).reset_index(drop=True)
    
sentences = sentences_df['sentences'].str.lower().tolist()
sentence_embeddings = model.encode(sentences)
question_1 = 'medical health care covid-19 coronavirus'
question_1_embedding = model.encode(question_1)

question_sentence_similarity_scores = []
for i in range(len(sentence_embeddings)):
    question_sentence_similarity_scores.append(scipy.spatial.distance.cdist([question_1_embedding[0]], [sentence_embeddings[i]], "cosine")[0])
    
sentences_df['cosine_score'] = question_sentence_similarity_scores
sentences_df.head()
for index, row in sentences_df[sentences_df['cosine_score'] > 1.3].iterrows():
    print(f"ID: {index}\nSentence: {row['sentences']}", '\n')
# Load Data and Library

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 
import seaborn as sns

from subprocess import check_output
! pip install wordcloud
from wordcloud import WordCloud, STOPWORDS

papers_df=papers_df.reset_index(drop=True)

META_DATA_PATH = "../input/CORD-19-research-challenge/"
meta_df = pd.read_csv(META_DATA_PATH + "metadata.csv")


# Data Source

df_sy=meta_df['source_x'].value_counts(dropna=False)
print(df_sy)

plt.barh(df_sy.index, df_sy.values, align='center', alpha=0.5,color='orange')
plt.ylabel('Quantity')
plt.title('Number of Literature by Sources')

plt.show()


# Literature Text

papers_df.shape

meta_df.shape

papers_df.shape[0]/meta_df.shape[0]

df_sy=meta_df['has_pmc_xml_parse'].value_counts(dropna=False)
print(df_sy)

plt.pie(df_sy.values,labels=df_sy.index,autopct='%1.1f%%',colors=['lightblue','orange'], shadow = True)
plt.title('Whether have PMC XML')
plt.axis('equal')
plt.show()


df_sy=meta_df['has_pdf_parse'].value_counts(dropna=False)
print(df_sy)

plt.pie(df_sy.values,labels=df_sy.index,autopct='%1.1f%%',colors=['orange','lightblue'], shadow = True)
plt.title('Whether have PDF')
plt.axis('equal')
plt.show()


df_sy=meta_df['full_text_file'].value_counts(dropna=False)
print(df_sy)
df_sy.index = pd.Series(df_sy.index).replace(np.nan, 'NaN')


plt.pie(df_sy.values,labels=df_sy.index,autopct='%1.1f%%',shadow = True)
plt.title('Full Text Source File')
plt.axis('equal')
plt.show()

plt.barh(df_sy.index, df_sy.values, align='center', alpha=0.5,color='orange')
plt.ylabel('Quantity')
plt.title('Number of Literatures with Full Text File')

plt.show()

### Comments:
### paper_df only have 38% data of metadata, should we use meta as well?
### PDF file have more info than PMC, seems like paper_df use PMC


# License

df_sy=meta_df['license'].value_counts(dropna=False)
print(df_sy)

plt.barh(df_sy.index, df_sy.values, align='center', alpha=0.5,color='orange')
plt.ylabel('Quantity')
plt.title('Number of Literatures by License')

plt.show()


# Authors

authors_count=meta_df['authors'].str.count(";")+1

df_sy=authors_count.isna().value_counts(dropna=False)
print(df_sy)

plt.pie(df_sy.values,labels=df_sy.index,autopct='%1.1f%%',colors=['lightblue','orange'], shadow = True)
plt.title('Whether Authors is NA')
plt.axis('equal')
plt.show()


plot_df_sy = authors_count.dropna()

print('Top Number of Authors:')
plot_df_sy.sort_values().tail()

sns.distplot(plot_df_sy, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3}
            ).set_title("Number of Authors Distribution")


# Affiliations

stopwords = set(STOPWORDS)

wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(papers_df['affiliations']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("wordcloud_affiliations.png", dpi=1024)


# Bibliography

wordcloud = WordCloud(
                          background_color='white',
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(papers_df['bibliography']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word_bibliography.png", dpi=1024)


# Time

df_sy=meta_df['publish_time'].str.slice(0, 4).value_counts(dropna=True)

df_sy=df_sy.sort_index()

from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 12), dpi=80, facecolor='w', edgecolor='k')

plt.barh(df_sy.index, df_sy.values)
plt.title('Number of Literatures by Year')
plt.ylabel('Year');
plt.show() 

### Comments:
### Most of the literatures are published in 2020, but there are still a large amount of papers published long time ago, should be cautious how reliable they are as for now


#journal
wordcloud = WordCloud(
                          background_color='white',
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(str(meta_df['journal']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
fig.savefig("word_journal.png", dpi=1024)

meta_df['journal'].value_counts(dropna=False).sort_values().tail()
### Comments:
### paper_df only have 38% data of metadata, we should use meta as well
### PDF file have more info than PMC, seems like paper_df only use PMC


### Comments:
### Most of the literatures are published in 2020, but there are still a large amount of papers published long time ago, should be cautious how reliable they are as for now


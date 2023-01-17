import warnings

warnings.simplefilter('ignore')

import pandas as pd  # For data handling

from time import time  # To time our operations

from collections import defaultdict  # For word frequency

import spacy 

import re 
df = pd.read_csv("../input/us-presidential-debatefinal-october-2020/Presidential_ debate_USA_ final_October_ 2020.csv")

df.drop(labels =['Unnamed: 0'],inplace = True,axis =1)

df.head()
df.TOPIC.value_counts()
df.speaker.value_counts()
grouped_df = df.groupby("speaker")       

grouped_df
grp_cnt = df['speaker'].value_counts()



# find who has used this word (people)

random_word = grouped_df['speech'].apply(lambda x: x.str.contains('people').sum()) 

word_df = pd.concat([grp_cnt,random_word],axis='columns',sort=False)

word_df
# rename columns

word_df.rename(columns={'speaker':'TotalSentsByThis_Speaker','speech':'numOfTimesWordUsed'},inplace=True)

# create a new column

word_df['Percent_of_word_usage'] = (word_df['numOfTimesWordUsed']/word_df['TotalSentsByThis_Speaker'])*100

word_df
word_df.sort_values(by = 'Percent_of_word_usage',ascending=False,inplace=True)

word_df.head()
# inspect a specific speaker that used this word

word_df.loc['TRUMP']
# spaCy based imports

import spacy

from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')
nlp.pipe_names
doc = nlp(df["speech"][4])

type(doc)
#Chunking

#spaCy automatically detects noun-phrases as well:



for chunk in doc.noun_chunks:

    print(chunk.text, chunk.label_, chunk.root.text)
#Dependency parsing

for token in doc:

    print("{0}/{1} <--{2}-- {3}/{4}".format(

        token.text, token.tag_, token.dep_, token.head.text, token.head.tag_))
from spacy import displacy

displacy.render(doc, style='dep', jupyter=True, options={'distance': 90})
from spacy.tokens import Doc

from nltk.sentiment.vader import SentimentIntensityAnalyzer

 

sentiment_analyzer = SentimentIntensityAnalyzer()

def polarity_scores(doc):

    return sentiment_analyzer.polarity_scores(doc.text)

 

Doc.set_extension('polarity_scores', getter=polarity_scores)

 

print(doc._.polarity_scores)
for token in doc:

  if token.like_num:

    print(token)
# creating a dictionary with parts of speech; corresponding token numbers.



all_tags = {token.pos: token.pos_ for token in doc}

print(all_tags)
# List to store number/cardinal

list_of_cardinals = []



# Appending entities which have the label 'CARDINAL' to the list

for entity in doc.ents:

  if entity.label_=='CARDINAL':

    list_of_cardinals.append(entity.text)



print(list_of_cardinals)
df['processed_speech'] = df['speech'].apply(lambda x: nlp(x))
df.head()
tokens = nlp(''.join(str(df.speech.tolist())))
#  extract the most common entities 

from collections import Counter

items = [x.text for x in tokens.ents]

Counter(items).most_common(20)
person_list = []

for ent in tokens.ents:

    if ent.label_ == 'PERSON':

        person_list.append(ent.text)

        

person_counts = Counter(person_list).most_common(20)

df_person = pd.DataFrame(person_counts, columns =['text', 'count'])
df_person.head()
import matplotlib.pyplot as plt 

df_person.plot.barh(x='text', y='count',

                  title="Person",

                  figsize=(10,8)).invert_yaxis()
#NORP type which recognizes nationalities, religious and political groups.

norp_list = []

for ent in tokens.ents:

    if ent.label_ == 'NORP':

        norp_list.append(ent.text)

        

norp_counts = Counter(norp_list).most_common(20)

df_norp = pd.DataFrame(norp_counts, columns =['text', 'count'])

df_norp.head()
df_norp.plot.barh(x='text', y='count',

                  title="Nationalities, Religious, and Political Groups",

                  figsize=(10,8)).invert_yaxis()


# let's use the loaded analyzer(NLTK)

analyzer = SentimentIntensityAnalyzer()



#Add VADER metrics to dataframe

df['neg'] = df['speech'].apply(lambda x:analyzer.polarity_scores(x)['neg'])

df['neu'] = df['speech'].apply(lambda x:analyzer.polarity_scores(x)['neu'])

df['pos'] = df['speech'].apply(lambda x:analyzer.polarity_scores(x)['pos'])

df['compound'] = df['speech'].apply(lambda x:analyzer.polarity_scores(x)['compound'])

df.head()
df.groupby('TOPIC')['compound'].describe()
df.groupby('speaker')['compound'].describe()
from textblob import TextBlob

#load the text into textblob

speech_blob = [TextBlob(text) for text in df['speech']]



#add the sentiment metrics to the dataframe

df['tb_Pol'] = [b.sentiment.polarity for b in speech_blob]

df['tb_Subj'] = [b.sentiment.subjectivity for b in speech_blob]



#show dataframe



df.head(3)
df.groupby('speaker')['tb_Pol'].describe()
df.groupby('speaker')['tb_Subj'].describe()
import scattertext as st
from scattertext import CorpusFromPandas, produce_scattertext_explorer

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from IPython.display import IFrame

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:98% !important; }</style>"))
df['parsed'] = df.speech.apply(nlp)  # parse for all speakers

# choose the candidates to compare

candidates = 'BIDEN', 'TRUMP'

df_candidates = df[df['speaker'].isin(candidates)]

df_candidates.head()
df_candidates['parsed'] = df_candidates.speech.apply(nlp)
# convert dataframe into Scattertext corpus

corpus_candidates = st.CorpusFromParsedDocuments(df_candidates, 

                                                 category_col='speaker',

                                                 parsed_col='parsed').build().remove_terms(ENGLISH_STOP_WORDS, ignore_absences=True)
term_freq_df = corpus_candidates.get_term_freq_df()

term_freq_df
term_freq_df['BIDEN Score'] = corpus_candidates.get_scaled_f_scores('BIDEN')

print(list(term_freq_df.sort_values(by='BIDEN Score', ascending=False).index[:10]))
term_freq_df['TRUMP Score'] = corpus_candidates.get_scaled_f_scores('TRUMP')

print(list(term_freq_df.sort_values(by='TRUMP Score', ascending=False).index[:10]))
# visualize term associations

html = produce_scattertext_explorer(corpus_candidates,

                                    category='BIDEN',

                                    category_name='Joe Biden',

                                    not_category_name='Trump',

                                    width_in_pixels=1000,

                                    #metadata= df_candidates['speaker'],

                                    metadata=corpus_candidates.get_df()['speaker'] + ': ' + corpus_candidates.get_df()['TOPIC'],

                                    minimum_term_frequency=3)

file_name = 'biden_trump_final_debate_terms.html'

open(file_name, 'wb').write(html.encode('utf-8'))

IFrame(src=file_name, width = 1000, height=700)
corpus_unigrams = st.CorpusFromParsedDocuments(

    df_candidates,

    category_col='speaker',

    parsed_col='parsed',

).build().get_unigram_corpus()



html = st.produce_scattertext_explorer(

    corpus_unigrams,

    category='BIDEN',

    not_category_name='Trump',

    minimum_term_frequency=0, 

    pmi_threshold_coefficient=0,

    width_in_pixels=1000, 

    metadata=corpus_unigrams.get_df()['speaker'] + ': ' + corpus_unigrams.get_df()['TOPIC'],

    transform=st.Scalers.dense_rank,

    use_full_doc=True



)

fn = 'BidenVERSUSTrump_final_debate_unigrams.html'

open(fn, 'wb').write(('<h2>Words used by Biden and Trump in the Final Debate</h2>' + html).encode('utf-8'))

IFrame(src=fn, width = 1300, height=700)
corpus_unigrams = st.CorpusFromParsedDocuments(

    df_candidates,

    category_col='speaker',

    parsed_col='parsed',

).build().get_unigram_corpus()



html = st.produce_scattertext_explorer(

    corpus_unigrams,

    category='BIDEN',

    not_category_name='Trump',

    minimum_term_frequency=0, 

    pmi_threshold_coefficient=0,

    width_in_pixels=1000, 

    metadata=corpus_unigrams.get_df()['speaker'] + ': ' + corpus_unigrams.get_df()['TOPIC'],

    transform=st.Scalers.dense_rank,

    use_full_doc=True



)

fn = 'BidenVERSUSTrump_final_debate_unigrams.html'

open(fn, 'wb').write(('<h2>Words used by Biden and Trump in the Final Debate</h2>' + html).encode('utf-8'))

IFrame(src=fn, width = 1300, height=700)
corpus_entities = st.CorpusFromParsedDocuments(

    df_candidates,

    category_col='speaker',

    parsed_col='parsed',

    feats_from_spacy_doc = st.SpacyEntities(entity_types_to_use=['PERSON', 'LOC', 'ORG', 'NAME'])

).build()



html = st.produce_scattertext_explorer(

    corpus_entities,

    category='BIDEN',

    not_category_name='Trump',

    minimum_term_frequency=0, 

    pmi_threshold_coefficient=0,

    width_in_pixels=1000, 

    metadata=corpus_entities.get_df()['speaker'] + ': ' + corpus_entities.get_df()['TOPIC'],

    transform=st.Scalers.dense_rank,

    use_full_doc=True



)

fn = 'bidenvstrump_final_debate_entities.html'

open(fn, 'wb').write(('<h2>Named Entities used by Biden and Trump in the Final Debate</h2>' + html).encode('utf-8'))

IFrame(src=fn, width = 1300, height=700)
corpus_moderator = st.CorpusFromParsedDocuments(

    df,

    category_col='speaker',

    parsed_col='parsed',

    feats_from_spacy_doc=st.FeatsFromSpacyDoc(use_lemmas=True)    

).build().get_stoplisted_unigram_corpus()



semiotic_square = st.SemioticSquare(

    corpus_moderator,

    category_a='BIDEN',

    category_b='TRUMP',

    neutral_categories=['WELKER'],

    scorer=st.RankDifference(),

    labels={'not_a_and_not_b': 'WELKER',

            'a_and_b': 'Candidates',

            'a_and_not_b': 'BIDEN',

            'b_and_not_a': 'TRUMP',

            'a':'',

            'b':'',

            'not_a':'',

            'not_b':''}

)
html = st.produce_semiotic_square_explorer(semiotic_square,

                                           category_name='Biden',

                                           not_category_name='Trump',

                                           x_label='Biden-Trump',

                                           y_label='Candidate-Moderator',

                                           num_terms_semiotic_square=5,

                                           width_in_pixels=800,

                                           neutral_category_name='WELKER',

                                           metadata=df['speaker'] + ' ' + df['TOPIC'])



fn = 'final_presidential_debate_MODERATOR_semiotic_.html'

open(fn, 'wb').write(html.encode('utf-8'))

IFrame(src=fn, width = 1300, height=700)
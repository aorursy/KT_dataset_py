import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/democratic-debate-transcripts-2020/debate_transcripts_v3_2020-02-26.csv', encoding='cp1252')

df['date'] = pd.to_datetime(df.date)

df.head()
df = df.loc[df.speaker.isin({'Joe Biden', 'Elizabeth Warren', 'Bernie Sanders', 'Pete Buttigieg', 'Amy Klobuchar', 'Michael Bloomberg', 'Tom Steyer', 'Tulsi Gabbard'})]

df.speaker.value_counts()
df.groupby(by='speaker').speaking_time_seconds.sum().plot.bar()

plt.show()
# Multi-Index on debate, candidate

debate_candidate_time = df.groupby(by=['date', 'speaker']).speaking_time_seconds.sum()

# Most recent debate

debate_candidate_time['2020-02-25']
# Multi-Index on candidate, debate

candidate_debate_time = df.groupby(by=['speaker', 'date']).speaking_time_seconds.sum()

candidate_debate_time
# Print Median Speaking Times

for candidate in df.speaker.unique():

    med = round(candidate_debate_time[candidate].median()/60)

    print(f'{candidate}: {med} minutes (median)')
# Plot Speaking Times Line Graph

plt.figure(figsize=(20,10))

for candidate in df.speaker.unique():

    candidate_debate_time[candidate].plot(label=candidate)

plt.legend()

plt.xlabel('date')

plt.ylabel('num seconds')

plt.title('Candidate Speaking Time over time')

plt.show()
import spacy



nlp = spacy.load('en_core_web_sm')
# I add additional stops that lead to lower quality topics

additional_stops = {'things', 'way', 'sure', 'thing', 'question', 'able', 'point', 'lot', 'time'}
from spacy.util import filter_spans





def _remove_stops(span):

    while span and span[0].pos_ not in {'ADJ', 'NOUN', 'PROPN'}:

        span = span.doc[span.start+1:span.end]

    return span





# Resource: 

# https://github.com/explosion/spacy/blob/master/examples/information_extraction/entity_relations.py

def merge_ents_and_nc(doc):

    spans = list(doc.ents) + list(doc.noun_chunks)

    spans = list(map(_remove_stops, spans))

    spans = filter_spans(spans)

    with doc.retokenize() as retokenizer:

        for span in spans:

            if span:

                # Added this in from the code - need to lemmatize better and keep ent types

                root = span.root

                attrs = {'LEMMA': span.text.lower(), 'POS': root.pos_, 'ENT_TYPE': root.ent_type_}

                retokenizer.merge(span, attrs=attrs)





def to_terms_list(doc):

    merge_ents_and_nc(doc)

    return [term.lemma_ for term in doc if len(term.lemma_) > 2 and\

             not term.is_stop and\

             term.pos_ in {'NOUN', 'PROPN', 'ADJ'} and\

             term.lemma_ not in additional_stops and\

             "crosstalk" not in term.lower_]
from spacy.tokens.doc import Doc



corpus = list(nlp.pipe(df.speech.values, n_threads=4))
terms = [to_terms_list(doc) for doc in corpus]

df['terms'] = terms
import itertools

from collections import Counter



def most_frequent(terms, k):

    flat_terms = list(itertools.chain(*terms))

    return Counter(flat_terms).most_common(k)
mf = df.groupby(by='speaker').terms.apply(lambda terms: most_frequent(terms, 10))

speakers = mf.index

top_terms = mf.values

for s, tt in zip(speakers, top_terms):

    print(f"{s}: {tt}\n")
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import NMF



# Useful hack: this dummy allows us to use our extracted terms from the methods above, overriding sklearn's tokenizer.

dummy = lambda x: x



vectorizer = TfidfVectorizer(max_df=0.5, min_df=10, preprocessor=dummy, tokenizer=dummy, ngram_range=(1,2))

features = vectorizer.fit_transform(terms)
N=8

tm = NMF(n_components=N)

doc_topic_matrix = tm.fit_transform(features)

topic_term_matrix = tm.components_

terms = vectorizer.get_feature_names()



# Array where rows are docs, columns are topics

print(doc_topic_matrix.shape)
def top_topic_terms(topic_term_matrix, i, n_terms=10):

    topic = topic_term_matrix[i]

    return [(terms[idx], topic[idx]) for idx in np.argsort(topic)[::-1][:n_terms]]
for i in range(N):

    print(i)

    print(top_topic_terms(topic_term_matrix, i))
names = {0: 'The American People',

         1: 'The President of the United States',

         2: 'Biden Telling us what he is saying is a matter of fact',

         3: 'The American Country',

         4: 'Global Diplomatic Issues',

         5: 'Social/Constitutional Rights',

         6: 'Healthcare and Medicare',

         7: 'Donald Trump'}
for i in range(N):

    col = f"topic_{i}"

    df[col] = doc_topic_matrix[:,i]

df.head()
def percent_comp(x):

    return x/x.sum()
df_speaker_topics = df.groupby(by=['speaker'])[df.columns[-N:]].sum()

df_speaker_topics
import matplotlib.pyplot as plt



topic_speaker_dist = df_speaker_topics.apply(percent_comp, axis=1)



for i, topic in enumerate(topic_term_matrix):

    print(names[i])

    print(' | '.join(terms[idx] for idx in np.argsort(topic)[::-1][:10]))

    topic_speaker_dist[f'topic_{i}'].plot.bar()

    plt.title(f'Topic: {names[i]}')

    plt.xlabel('Candidate')

    plt.ylabel('Percent of Topic')

    plt.show()
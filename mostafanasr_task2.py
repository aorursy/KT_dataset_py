# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

dirctory_name = '/kaggle/input/CORD-19-research-challenge/'

# for dirname, _, filenames in os.walk(dirctory_name):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
metadata = pd.read_csv(dirctory_name+"metadata.csv")

metadata.head()
metadata.info()
records_with_abstract = metadata[metadata['abstract'].notna()]

records_with_abstract.count()
records_with_abstract['abstract'].describe(include='all')
mask = records_with_abstract['abstract'].str.len() < 20

print(records_with_abstract.loc[mask][["abstract"]])
records_with_abstract['abstract'].describe(include='all')
mask20 = records_with_abstract['abstract'].str.len() > 50

records_with_abstract = records_with_abstract.loc[mask20]

records_with_abstract.head()
records_with_abstract['abstract'].describe(include='all')


# records_with_abstract.drop_duplicates(['abstract'], inplace=True)

# records_with_abstract['abstract'].describe(include='all')

# copyDF = records_with_abstract.copy()

# copyDF.drop_duplicates(['abstract'], inplace=True)

duplicateRowsDF = records_with_abstract[records_with_abstract.duplicated(['abstract'])]

duplicateRowsDF.tail()
records_with_abstract.drop_duplicates(['abstract'], inplace=True)

records_with_abstract['abstract'].describe(include='all')
records_with_abstract.tail(30)
id_title_abstract_docs = []

i = 0

for index,doc_meta in records_with_abstract.iterrows():

    data = [i,doc_meta["title"],doc_meta["abstract"]]

    i+=1

    id_title_abstract_docs.append(data)
import spacy

from spacy.matcher import Matcher



nlp = spacy.load("en_core_web_lg")


matcher = Matcher(nlp.vocab)

# Add match ID "HelloWorld" with no callback and one pattern

pattern0 = [{"LEMMA": {"IN": ["transmission", "transmissible","severity", "disease","fatal","fatality","infection","risk factors","viral","smoking"]}}]

pattern1 = [{"LOWER": "novel"}, {"LOWER": "coronavirus"}]

pattern3 = [{"LOWER": "pulmonary"}, {"LOWER": "disease"}]

pattern4 = [{"LOWER": "pulmonary"}]





matcher.add("task2_Related", None, pattern0,pattern1,pattern3,pattern4)



corona_docs = []

for id_title_abstract_doc in id_title_abstract_docs:

    doc = nlp(id_title_abstract_doc[2])

    matches = matcher(doc)

    for match_id, start, end in matches:

        string_id = nlp.vocab.strings[match_id]  # Get string representation

        span = doc[start:end]  # The matched span

        if id_title_abstract_doc[0] not in corona_docs:

            corona_docs.append(id_title_abstract_doc[0])

#             print(id_title_abstract_doc[0] ,match_id, string_id, start, end, span.text)

        

        
import os

import json



with open("/kaggle/working/Myabstracts.json", 'w') as f:

    json.dump(corona_docs, f)



#reading the saved file

testMyabstracts = []

with open("/kaggle/working/Myabstracts.json", 'r') as f:

    testMyabstracts = json.load(f)

corona_docs = testMyabstracts
print(len(corona_docs))
smoking_query = nlp("smoking makes coronavirus worse")

similar_papers = []

similarity_Array =[]

similarity_80 = []

for curr_doc in corona_docs:

    curr_abstract = nlp(id_title_abstract_docs[curr_doc][2])

    similarity = curr_abstract.similarity(smoking_query)

    if (similarity > 0.7):

        similarity_Array.append(similarity)

        similar_papers.append(curr_doc)

        print("a similarity between smoking and "+ str(curr_doc) + str(similarity))

    if (similarity > 0.8):

        similarity_80.append(curr_doc)

        
print(similar_papers)
print(len(similarity_Array),len(similarity_80))
with open("/kaggle/working/similarToOurQuery.json", 'w') as f:

    json.dump(similar_papers, f)
#reading the saved file

readSimilarPapers = []

with open("/kaggle/working/similarToOurQuery.json", 'r') as f:

    readSimilarPapers = json.load(f)

similar_papers = readSimilarPapers
for relatedDoc in similar_papers:

    print(id_title_abstract_docs[relatedDoc][2]+ "\n")
from spacy.lang.en.stop_words import STOP_WORDS

from string import punctuation
for relatedDoc in similar_papers:

    doc = id_title_abstract_docs[relatedDoc][2]

    extra_words=list(STOP_WORDS)+list(punctuation)+['\n']

    nlp=spacy.load('en')

    docx = nlp(doc)

    all_words=[word.text for word in docx]

    Freq_word={}

    for w in all_words:

        w1=w.lower()

        if w1 not in extra_words and w1.isalpha():

            if w1 in Freq_word.keys():

                Freq_word[w1]+=1

            else:

                Freq_word[w1]=1

    val=sorted(Freq_word.values())

    max_freq=val[-3:]

    print("Topic of document given :-")

    for word,freq in Freq_word.items():

        if freq in max_freq:

             print(word ,end=" ")

        else:

              continue

    for word in Freq_word.keys():

        Freq_word[word] = (Freq_word[word]/max_freq[-1])

    sent_strength={}

    

    for sent in docx.sents:

        for word in sent :

            if word.text.lower() in Freq_word.keys():

                if sent in sent_strength.keys():

                     sent_strength[sent]+=Freq_word[word.text.lower()]

                else:

                     sent_strength[sent]=Freq_word[word.text.lower()]

            else: 

                continue



    top_sentences=(sorted(sent_strength.values())[::-1])

    top30percent_sentence=int(0.3*len(top_sentences))

    top_sent=top_sentences[:top30percent_sentence]

    summary=[]

    for sent,strength in sent_strength.items():

        if strength in top_sent:

            summary.append(sent)

        else:

            continue



    for i in summary:

        print(i,end=" ")

    print("\n")
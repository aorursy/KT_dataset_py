import pandas as pd

metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')

metadata.columns
metadata['WHO #Covidence'].value_counts()
metadata.has_full_text.value_counts()
pub_with_full_text_metadata = metadata.loc[metadata.has_full_text==True,]

pub_with_full_text_metadata.shape
journal_names=pub_with_full_text_metadata.journal.unique()

len(journal_names)
journal_names
#pub_with_full_text_metadata[pub_with_full_text_metadata.groupby(11)[11].transform('count')>100,]

#pub_with_full_text_metadata[pub_with_full_text_metadata['journal'].isin(pub_with_full_text_metadata['journal'].value_counts().get(100).loc[lambda x : x].index)]

#pub_with_full_text_metadata[pub_with_full_text_metadata.groupby('journal')['journal'].transform('count')>100,]

pub_with_full_text_metadata_common_journals=pub_with_full_text_metadata.groupby('journal').filter(lambda x: len(x) >= 200)

pub_with_full_text_metadata_common_journals.journal.value_counts()
good_journals=['The Lancet','Emerg Infect Dis','PLoS Pathog']

pub_in_good_journals_with_full_text_metadata = pub_with_full_text_metadata[pub_with_full_text_metadata.journal.isin(good_journals)]
pub_in_good_journals_with_full_text_metadata.shape
import json

with open('/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/e49106784cd05d905c10780cc5dbe2a10a6badb7.json') as jsonfile:

    parsed = json.load(jsonfile)

print(json.dumps(parsed, indent=2, sort_keys=True))
import os

articles = {}

stat = { }

for dirpath, subdirs, files in os.walk('/kaggle/input'):

    for x in files:

        if x.endswith(".json"):

         articles[x] = os.path.join(dirpath, x)
articles['25621281691205eb015383cbac839182b838514f.json']
from tqdm import tqdm   

literature = {}

for index, row in tqdm(pub_in_good_journals_with_full_text_metadata.iterrows(), total=pub_in_good_journals_with_full_text_metadata.shape[0]):

    sha = str(row['sha'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            found = False

            with open(articles[sha]) as f:

                data = json.load(f)

                for key in ['body_text']:

                    if found == False and key in data:

                        for content in data[key]:

                            if sha in literature:

                                oldcontent = literature[sha]

                                newcontent = content['text']

                                text = oldcontent + newcontent

                            else:

                                text = content['text']

                            literature[sha]=text                                

        except KeyError:

            pass
!pip install git+https://github.com/boudinfl/pke.git
!python -m nltk.downloader stopwords

!python -m nltk.downloader universal_tagset

!python -m spacy download en
pub_in_good_journals_with_full_text_metadata_with_keyphrases = pub_in_good_journals_with_full_text_metadata.reindex(columns=[*pub_in_good_journals_with_full_text_metadata.columns.tolist(),'Keyphrase_1', 'Keyphrase_2','Keyphrase_3','Keyphrase_4','Keyphrase_5','Keyphrase_6','Keyphrase_7','Keyphrase_8','Keyphrase_9','Keyphrase_10'], fill_value='')
from io import StringIO

import string

import pke

from nltk.corpus import stopwords

stoplist = list(string.punctuation)

stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']

stoplist += stopwords.words('english')

for index, row in tqdm(pub_in_good_journals_with_full_text_metadata_with_keyphrases.iterrows(), total=pub_in_good_journals_with_full_text_metadata_with_keyphrases.shape[0]):

    sha = str(row['sha'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            body_text = literature[sha]

            if(len(body_text.split())>300):

                f = StringIO(literature[sha])

                extractor = pke.unsupervised.TopicRank()

                extractor.load_document(input=f, language='en',normalization='stemming')

                # keyphrase candidate selection, in the case of TopicRank: sequences of nouns

                # and adjectives (i.e. `(Noun|Adj)*`)

                extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ','ADV','VERB'},stoplist=stoplist)

                extractor.candidate_filtering(maximum_word_number=3,minimum_word_size=5,minimum_length=20,only_alphanum=True)

                # candidate weighting, in the case of TopicRank: using a random walk algorithm

                extractor.candidate_weighting(threshold=0.9,method='ward',heuristic='frequent')

                # N-best selection, keyphrases contains the 10 highest scored candidates as (keyphrase, score) tuples

                keyphrases = extractor.get_n_best(n=10,redundancy_removal=True)

                indx=0

                for phrase in keyphrases:

                    indx = indx + 1

                    colname = 'Keyphrase_'+str(indx)

                    pub_in_good_journals_with_full_text_metadata_with_keyphrases.loc[index, colname] = phrase[0]

        except KeyError:

            pass
pub_in_good_journals_with_full_text_metadata_with_keyphrases.head()
pub_in_good_journals_with_full_text_metadata_with_keyphrases.to_csv('metadata_with_keyphrases_sample.csv')
import os

os.chdir(r'/kaggle/input/keyphrase-extraction-output')

from IPython.display import FileLink

FileLink(r'metadata_with_keyphrases_sample.csv')
#metadata_with_keyphrases=pub_in_good_journals_with_full_text_metadata_with_keyphrases

metadata_with_keyphrases = pd.read_csv('/kaggle/input/keyphrase-extraction-output/metadata_with_keyphrases_sample.csv')

relevant_publications_metadata=metadata_with_keyphrases.loc[lambda x: ( 

                                        x.Keyphrase_1.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_2.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_3.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_4.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_5.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_6.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_7.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_8.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_9.str.contains(r'\b(?:acute|severe)\b',regex=True) | 

                                        x.Keyphrase_10.str.contains(r'\b(?:acute|severe)\b',regex=True))]



relevant_publications_metadata=relevant_publications_metadata.loc[lambda x: ( 

                                        x.Keyphrase_1.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_2.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_3.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_4.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_5.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_6.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_7.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_8.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_9.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True) | 

                                        x.Keyphrase_10.str.contains(r'\b(?:respiratory infections|respiratory illness|coronavirus|coronaviruses|Coronavirus)\b',regex=True))]



relevant_publications_metadata=relevant_publications_metadata.loc[lambda x: ( 

                                        x.Keyphrase_1.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_2.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_3.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_4.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_5.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_6.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_7.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_8.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_9.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True) | 

                                        x.Keyphrase_10.str.contains(r'\b(?:infant|neonatal|children)\b',regex=True))]

relevant_publications_metadata.shape
for index, row in tqdm(relevant_publications_metadata.iterrows(), total=relevant_publications_metadata.shape[0]):

    sha = str(row['sha'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            print(relevant_publications_metadata.title[index])

            print('\n')

        except KeyError:

            pass
relevant_publications_metadata=metadata_with_keyphrases.loc[lambda x: ( 

                                        x.Keyphrase_1.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_2.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_3.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_4.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_5.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_6.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_7.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_8.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_9.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True) | 

                                        x.Keyphrase_10.str.contains(r'\b(?:outbreak|Outbreak|outbreaks|pandemic|epidemic)\b',regex=True))]

for index, row in tqdm(relevant_publications_metadata.iterrows(), total=relevant_publications_metadata.shape[0]):

    sha = str(row['sha'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            print(relevant_publications_metadata.title[index])

            print('\n')

        except KeyError:

            pass
relevant_publications_metadata=metadata_with_keyphrases.loc[lambda x: ( 

                                        x.Keyphrase_1.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_2.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_3.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_4.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_5.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_6.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_7.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_8.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_9.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True) | 

                                        x.Keyphrase_10.str.contains(r'\b(?:replication|mutation|infection|replications|mutations|infections)\b',regex=True))]

for index, row in tqdm(relevant_publications_metadata.iterrows(), total=relevant_publications_metadata.shape[0]):

    sha = str(row['sha'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            print(relevant_publications_metadata.title[index])

            print('\n')

        except KeyError:

            pass
relevant_publications_metadata=metadata_with_keyphrases.loc[lambda x: ( 

                                        x.Keyphrase_1.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_2.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_3.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_4.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_5.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_6.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_7.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_8.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_9.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True) | 

                                        x.Keyphrase_10.str.contains(r'\b(?:community|transmission|transmissions|spread)\b',regex=True))]

for index, row in tqdm(relevant_publications_metadata.iterrows(), total=relevant_publications_metadata.shape[0]):

    sha = str(row['sha'])

    if sha != 'nan':

        sha = sha + '.json';

        try:

            print(relevant_publications_metadata.title[index])

            print('\n')

        except KeyError:

            pass      
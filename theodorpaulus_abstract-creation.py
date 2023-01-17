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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from nltk import pos_tag
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer,WordNetLemmatizer
import json 
from numpy import random

pst=PorterStemmer()
wnl=WordNetLemmatizer()
risk_factors={
    "hypertension",
    "diabetes",
    "male",
    "heart disease",
    "COPD",
    "Smoking Status",
    "Age",
    "cerebrovascular disease",
    "cardiovascular disease",
    "cancer",
    "respiratory system disease",
    "kidney disease",
    "respiratory disease",
    "drinking",
    "overweight",
    "obese",
    "chronic liver disease"
}

material_factors={    
    'viral shedding',
    'stool',
    'nasopharynx',
    'urine',
    'blood',
    'virus persist',
    'inanimate surfaces',
    'virus remain viable',
     'persistence on different materials',
    'adhesion to hydrophilic surfaces',
    'adhesion to hydrophobic surfaces',
    'cleaning agents',
    'decontamination',
    'physical'

}
Material={
    'steel',
    'copper',
    'plastic',
    'glass',
    'wood',
    'paper',
    'cardboard',
    'textile',
    'fabric',
    'bed',
    'sheet',
    'rail',
    'door',
    'knob',
    'lever'
}


Study_design={'Systematic review','meta-analysis',
              'Prospective observational study',
              'Retrospective observational study',
              'Cross-sectional study',
              'Expert review',
              'Editorial',
              'Ecological regression',
              'Simulation'
             }

key_words={'cases',
           'pneumonia',
           'bronchitis',
           'pulmonary disease',
           'source disease',
           'virus',
           'virus caused'}

key_words.update(material_factors)
key_words.update(Study_design)
#key_words.update(risk_factors)
#key_words=pst.stem(key_words)
j=0
key_words_lemma={'%'}
for x in key_words:
    key_words_lemma.add(wnl.lemmatize(x))
    j=j+1

print(key_words_lemma)
metadata_df=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')
metadata_df.head()
print(metadata_df.shape)
metadata_df.dropna(how='all',subset=['pdf_json_files','pmc_json_files'],inplace=True)
i=0
no_abstracts_article=metadata_df[metadata_df.abstract.isna()].index

for s in metadata_df[metadata_df.abstract.notna()].index:
    if len(metadata_df.abstract.loc[s])<250:        
        break
        no_abstracts_article.append(s)
        i=i+1
print(no_abstracts_article)
def create_abstract(index_loc,key_words_lemma,key_word_weight=200,sentence_length_weight=0,threshold_value_sent=200):
    if (metadata_df['pdf_json_files'].notna().loc[index_loc]):
        path='../input/CORD-19-research-challenge/'+metadata_df.loc[index_loc]['pdf_json_files']
    else:
        if (metadata_df['pmc_json_files'].notna().loc[index_loc]):
            path='../input/CORD-19-research-challenge/'+metadata_df.loc[index_loc]['pmc_json_files']
        else:
            path=-1
            
    if (path is not -1) :
        file_source=open(path,'r')
        string_source=""
        for x in file_source:
            string_source=string_source+x
        #load json file
        json_obj_source=json.loads(string_source)
        stoplist=stopwords.words('english')
        new_abstr=""
        j=0
        for x in json_obj_source['body_text']:
            sentences=sent_tokenize(x['text'])
            score_sentences=[]
            for y in sentences:
                #y=pst.stem(y)
                y=wnl.lemmatize(y)
                s=[word for word in y.split() if word not in [stoplist,'The','the','of','in','is','an','as','it','a','to']]
                score=0
                for z in s:
                    if z in key_words_lemma:
                        score=score+key_word_weight        
                score=score+len(s)*sentence_length_weight
                score_sentences.append(score)
            i=0
            for y in sentences:        
                if (score_sentences[i]>=threshold_value_sent):
                    new_abstr=new_abstr+y
                i=i+1
            #print(new_abstr)
            metadata_df.at[index_loc,'abstract']=new_abstr
        return 1
    else:
        #not valid path
        return 0
        


for index_loc in no_abstracts_article:
    create_abstract(index_loc,key_words_lemma)
    #metadata_df.at[index_loc,'abstract']=new_abstr
    print(metadata_df.at[index_loc,'abstract'])
metadata_df.head()
metadata_df['Study']="-"
metadata_df['Material']="-"
metadata_df['Method']="-"
metadata_df.columns
for i in metadata_df.index:
    title=metadata_df['title'].loc[i]
    for kw in Study_design:
        if (type(title)=='str'):
            if (title.find(kw) is not -1):
                metadata_df['Study'].loc[i]= kw
        else:
            abstract=metadata_df['abstract'].loc[i]
            if (type(abstract)=='str'):
                if (abstract.find(kw) is not -1):
                    metadata_df['Study'].loc[i]= kw
    
    for kw in Material:
        if (type(title)=='str'):
            if (title.find(kw) is not -1):
                metadata_df['Material'].loc[i]= kw
        else:
            abstract=metadata_df['abstract'].loc[i]
            if (type(abstract)=='str'):
                if (abstract.find(kw) is not -1):
                    metadata_df['Material'].loc[i]= kw
            
(metadata_df['Study']!="-").any()
(metadata_df['Material']!="-").any()
metadata_df.to_csv("metadata_df_abstracts4all.csv")
#metadata_df=pd.read_csv("metadata_df_abstracts4all.csv")
text2test="BACKGROUND: AlkB-like proteins are members of the 2-oxoglutarate- and Fe(II)-dependent oxygenase superfamily. In Escherichia coli the protein protects RNA and DNA against damage from methylating agents. 1-methyladenine and 3-methylcytosine are repaired by oxidative demethylation and direct reversal of the methylated base back to its unmethylated form. Genes for AlkB homologues are widespread in nature, and Eukaryotes often have several genes coding for AlkB-like proteins. Similar domains have also been observed in certain plant viruses. The function of the viral domain is unknown, but it has been suggested that it may be involved in protecting the virus against the post-transcriptional gene silencing (PTGS) system found in plants. We wanted to do a phylogenomic mapping of viral AlkB-like domains as a basis for analysing functional aspects of these domains, because this could have some relevance for understanding possible alternative roles of AlkB homologues e.g. in Eukaryotes. RESULTS: Profile-based searches of protein sequence libraries showed that AlkB-like domains are found in at least 22 different single-stranded RNA positive-strand plant viruses, but mainly in a subgroup of the Flexiviridae family. Sequence analysis indicated that the AlkB domains probably are functionally conserved, and that they most likely have been integrated relatively recently into several viral genomes at geographically distinct locations. This pattern seems to be more consistent with increased environmental pressure, e.g. from methylating pesticides, than with interaction with the PTGS system. CONCLUSIONS: The AlkB domain found in viral genomes is most likely a conventional DNA/RNA repair domain that protects the viral RNA genome against methylating compounds from the environment."
sent2test=sent_tokenize(text2test)
print(sent2test)

tokenized_sentences=[word_tokenize(sentence) for sentence in sent2test]
#print(tokenized_sentences)
tagged_sent=[pos_tag(sentence) for sentence in tokenized_sentences]
from nltk import ne_chunk
#chunk_sent=[ne_chunk(sentence) for sentence in tagged_sent]
#print(chunk_sent)
for sentence in tagged_sent:
    print(ne_chunk(sentence))
    break
import nltk
chart_parser=nltk.parse.chart.Chart(tokenized_sentences)
chart_parser.leaves()

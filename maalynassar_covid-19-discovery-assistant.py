import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
import re
import os
import string
import random
import json
import sys
import spacy #pip install -U spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")#python -m spacy download en_core_web_sm
#sci = spacy.load("en_core_sci_md")
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors
import requests
from pathlib import Path
from scipy.spatial import distance
from ast import literal_eval
#med7 = spacy.load("en_core_med7_lg")#https://github.com/kormilitzin/med7
### paths ###
word2vecdir = '/kaggle/input/covidw2v/covid_w2v/' #word2vec dir
csvdir = '/kaggle/input/covid19texts/'# pubmed open access csv dir
jsondirs = ['/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/']# kaggle common_use pmc dir
### Retrieving Kaggle data PMC IDs from common_use_subset directory ###
kpmcids =[]
for jsondir in jsondirs:
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:       
            if filename.endswith('.json'):
                jfile = os.path.join(dirname, filename)
                #print(jfile)
                with open(os.path.join(jsondir, jfile)) as json_file:
                    vjson = json.load(json_file)
                    if 'PMC'in vjson['paper_id']:
                        kpmcids.append(vjson['paper_id'])           
                        
### LOADING OPEN ACCESS PUBMED TEXTS ###
csvf = os.path.join(csvdir,'covid-19-texts.csv')
print(csvf)
data = pd.read_csv(csvf)
data.fillna(0,inplace=True)
data
### filtering pubmed texts to include those from kaggle only [N=1265] ###
data = data[data['PMC'].isin(kpmcids)]
pmcs = list(data['PMC'])
data
### filter data to those published in 2019 and 2020 only ###
#cdates = ['2019','2020']
#data['cDATE'] = [d for d in data['DATE']]
#data['cDATE'] = ['Y' if '2020' in d or '2019' in d else 'N' for d in data['cDATE']]
#data = data[data['cDATE']!='N']
### filtering data to those containing corona in their abstracts ###
data['cTOPIC'] = [d for d in data['TITLE']]
data['cTOPIC'] = ['Y' if 'corona' or 'sar-cov' or 'covid' in str(d) else 'N' for d in data['ABS']]
data = data[data['cTOPIC']!='N']
pmcs = list(data['PMC'])
data
### generating sentences from text ###
stopwords = nltk.corpus.stopwords.words('english')
all_sents_words = []
all_sents = []
for t, txt in enumerate(data['text']):    
    sents = nltk.sent_tokenize(txt)      
    for sent in sents:
        words = nltk.word_tokenize(sent)                 
        words = [word for word in words if word.lower() not in stopwords and word.lower() not in string.punctuation]
        all_sents_words.append(words)
        all_sents.append([pmcs[t],sent])
### Bag of words covering kaggle challenge topics ###
cents = {}
cents['studies'] = ['naproxen clarithromycin minocyclinethat','viral inhibitor']
cents['ADE'] = ['Antibody-Dependent Enhancement (ADE)']
cents['animal'] = ['animal','model']
cents['therapy'] = ['IL-6 camostat mesylate chloroquine hydroxychloroquine lopinavir ritonavir ivermectin remdesivir heparin','antiviral agent','monclonal antibodies viral','convalscent plasma']
cents['decision'] = ['therapeutics production','decision making','model']
cents['vaccine'] = ['vaccine','universal','covid-19']
cents['prophylaxis'] = ['prophylaxis','health care workers']
cents['risk'] = ['adverse effects','potential complication','vaccine']
cents['assay'] = ['assays','immune response','vaccine development','animal model','therapeutics']
cents['transmission'] = ['transmission']
cents['pathogenesis'] = ['spike glycoprotein','Antigen','Receptor']
cents['symptoms'] = ['symptoms']
cents['immunity'] = ['Cytokines Interleukins Interferon Antibodies CD4 CD8','Immunity']
cents['diagnosis'] = ['ELISA RT-PCR Western blot immunofluorescence radiography','diagnostic','kits','autopsy','assay','lung-derived cells','serodiagnosis','serology','virus isolation','specimens','cell culture']

### topics of interest ###
catgs = list(cents.keys())
### Word embeddings models ###
fast_embeds = ['ncbi_corona_TAGD_WV_fasttext_200_5.model']
fast_ttls = ['fasttext_200_5']
vdfgs = {}
### scanning discussion sections for topic-related sentences ###
for n,m in enumerate(fast_embeds):    
    model = Word2Vec.load(word2vecdir+m)
    wv = KeyedVectors.load(word2vecdir+m)
    #vsdfs =[]    
    for j, v in enumerate(catgs): 
        #print(j)
        #print(v)
        vcol1 = []
        vcol2 = []
        vcol3 = []
        vcol4 = []
        vcol5 = []
        vcol6 = []
                
        ### generating topic embeddings ###
        syns = cents[catgs[j]]         
        syns_avec = wv[' '.join(cents[catgs[j]])] # topic words representations (embeddings)
        
        ### retrieving topic-related sentences ###
        vsdsts = []        
        vwords = []
        for s,sent in enumerate(all_sents_words):            
            asent_vec = wv[' '.join(sent)] # sentence words representations (embeddings)
            sdst = np.around(distance.cosine(syns_avec,asent_vec),decimals=2) # 
            vsdsts.append(sdst)
            wdsts = [distance.cosine(wv[cents[catgs[j]][0]],wv[word]) for word in sent]
            sort_wdsts_keys = np.argsort(wdsts)
            sort_wdsts ={sent[i]:np.around(wdsts[i],decimals=2) for i in sort_wdsts_keys if np.around(wdsts[i],decimals=2)<0.5}   
            vwords.append(sort_wdsts)
        sort_vsdsts_keys = np.argsort(vsdsts)
        for i in sort_vsdsts_keys:
            if vsdsts[i]<0.5:
                pmcid = all_sents[i][0]
                vcol1.append(pmcid)
                vcol2.append(all_sents[i][1])
                vcol3.append(vsdsts[i])
                vcol4.append(vwords[i])
                vcol5.append(list(data[data['PMC'] == pmcid]['DATE'])[0]) if list(data[data['PMC'] == pmcid]['DATE'])!=[] else vcol5.append('')
                vcol6.append(list(data[data['PMC'] == pmcid]['AFF'])[0]) if list(data[data['PMC'] == pmcid]['AFF'])!=[] else vcol6.append('')
        vdf = pd.DataFrame([vcol1,vcol2,vcol3,vcol4,vcol5,vcol6]).transpose()
        vdf.columns = ['PMC-ID','SENT','COS-SIM','ENTITY','DATE','AFF']
        vdf.columns = ['PMC-ID','SENT','COS-SIM','ENTITY','DATE','AFF']
        vdfg1 = vdf.groupby(['PMC-ID'],sort=False)['SENT'].apply(list).reset_index()
        vdfg2 = vdf.groupby(['PMC-ID'],sort=False)['DATE'].apply(list).reset_index()
        vdfg3 = vdf.groupby(['PMC-ID'],sort=False)['ENTITY'].apply(list).reset_index()
        vdfg4 = vdf.groupby(['PMC-ID'],sort=False)['AFF'].apply(list).reset_index()
        vdfg = vdfg1.join(vdfg2['DATE']).join(vdfg3['ENTITY']).join(vdfg4['AFF'])
        vdfg['DATE'] = [d[0] for d in vdfg['DATE']] # cleaning date column
        vdfg['AFF'] = [d[0] for d in vdfg['AFF']] # cleaning affiliation column
        vdfg['ENTITY'] = [list(np.unique([enk for ents in entss for enk in list(ents.keys())])) for entss in vdfg['ENTITY']] # cleaning entities column        
        vdfgs[v] = vdfg            
### Animal models tried with coronaviruses ###
for n,m in enumerate(vdfgs['animal']):
    row = vdfgs['animal'].iloc[n+1]
    print(row['PMC-ID'])
    print(' ')
    print(' ')
    print(row['SENT'][0:5])
    print(' ')
    print('###################################')
    print(' ')
### Therapeutics tested with coronaviruses ###
for n,m in enumerate(vdfgs['therapy']):
    row = vdfgs['therapy'].iloc[n+1]
    print(row['PMC-ID'])
    print(' ')
    print(' ')
    print(row['SENT'][0:5])
    print(' ')
    print('###################################')
    print(' ')
### ADE associated with coronaviruses ###
for n,m in enumerate(vdfgs['ADE']):
    row = vdfgs['ADE'].iloc[n+1]
    print(row['PMC-ID'])
    print(' ')
    print(' ')
    print(row['SENT'][0:5])
    print(' ')
    print('###################################')
    print(' ')
### Immunity responses to coronaviruses ###
for n,m in enumerate(vdfgs['immunity']):
    row = vdfgs['immunity'].iloc[n]
    print(row['PMC-ID'])
    print(' ')
    print(row['SENT'][0:5])
    print(' ')
    print('###################################')
    print(' ')

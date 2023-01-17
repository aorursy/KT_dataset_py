!pip install --quiet scispacy
!pip install --quiet https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz
import pandas as pd
from io import StringIO
riskfac = StringIO("""Factor;Description
    Pulmonary;Smoking, preexisting pulmonary disease
    Infection;Coinfections determine whether coexisting respiratory or viral infections make the virus more transmissible or virulent and other comorbidities
    Birth;Neonates and pregnant women
    Socio-eco;Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences
    Transmission;Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
    Severity;Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
    Susceptibility;Susceptibility of populations
    Mitig-measures;Public health mitigation measures that could be effective for control
    """)

rf_base = pd.read_csv(riskfac, sep= ";")
rf_base
# exporting factors and description to save it. rf_base.to_csv(r'/2020-03-13/rf_base.csv', index = False).
rf_base.to_csv('rf_base.csv',index=False)
data = pd.read_csv('../working/rf_base.csv', delimiter=',', header=None, skiprows=1, names=['Factor','Description'])

descp = data[:8][['Description']];
descp['index'] = descp.index
descp
fact_name = data[:8][['Factor']];
fact_name['index'] = fact_name.index
fact_name
import scispacy
import spacy # needs to update spacy to load SciSpacy model.
from spacy import displacy

## need to install "en_core_sci_md" model https://allenai.github.io/scispacy/

nlp = spacy.load('en_core_sci_md') # "en_core_sci_md" larger biodmedical vocab. word vector

def patternizing(dataF):
    for i in range(8):
        theme_sample = dataF[dataF['index'] == i].values[0][0]
        
        text = theme_sample
        # print(theme_sample)

        doc = nlp(text)
       
        # print(list(doc.sents))
        # print(doc.ents)
        
        displacy.render(next(doc.sents), style='ent', jupyter=True)
patternizing(descp)
import spacy
nlp = spacy.load("en_core_sci_md")

def pRnize(dataF, indice):
    
    mastlist = []
    """
    n_cov2 = ['covid19', 'covid-19',
              'Covid19', 'Covid-19',
              'COVID19', 'COVID-19',
              'Sars-Cov-2', 'Sars-CoV-2', 'Sars-COV-2', 'Sars-cov-2',
              'SARS-Cov-2', 'SARS-CoV-2', 'SARS-COV-2', 'SARS-cov-2',
              'sars-Cov-2', 'sars-CoV-2', 'sars-COV-2', 'sars-cov-2',
              'Sars Cov-2', 'Sars CoV-2', 'Sars COV-2', 'Sars cov-2',
              'SARS Cov-2', 'SARS CoV-2', 'SARS COV-2', 'SARS cov-2',
              'sars Cov-2', 'sars CoV-2', 'sars COV-2', 'sars cov-2',
              'Sars-Cov 2', 'Sars-CoV 2', 'Sars-COV 2', 'Sars-cov 2',
              'SARS-Cov 2', 'SARS-CoV 2', 'SARS-COV 2', 'SARS-cov 2',
              'sars-Cov 2', 'sars-CoV 2', 'sars-COV 2', 'sars-cov 2',
              'Sars Cov 2', 'Sars CoV 2', 'Sars COV 2', 'Sars cov 2',
              'SARS Cov 2', 'SARS CoV 2', 'SARS COV 2', 'SARS cov 2',
              'sars Cov 2', 'sars CoV 2', 'sars COV 2', 'sars cov 2',
              'Sars Cov2', 'Sars CoV2', 'Sars COV2', 'Sars cov2',
              'SARS Cov2', 'SARS CoV2', 'SARS COV2', 'SARS cov2',
              'sars Cov2', 'sars CoV2', 'sars COV2', 'sars cov2',]
    """
    for i in range(8):
        factor = []
        theme_sample = dataF[dataF['index'] == i].values[0][0]

        text = theme_sample

        doc = nlp(text) 
        
        for item in doc.ents:
            vocab = str(item).lower().strip('()')
            factor.append(vocab)
        
        #for name in n_cov2:
         #   factor.append(name)
        mastlist.append(factor)
    return mastlist[indice]

# To test unquote
#pRnize(descp, 0)
def key_per_theme(dataF, word):
    dic = {}
    for i in range(8):
        factor = dataF[dataF['index'] == i].values[0][0]
        wordy = pRnize(word, i)
    
        dic[factor.strip()] = wordy
    return dic
key_per_theme(fact_name, descp)
import os
import json
import glob
# json file access function
def data_access(path):
    d_acc = {}
    for i in glob.glob(path):
        # link = os.path.normpath(i)
        # print(link)
        
        # loading json file function
        with open(os.path.normpath(i)) as json_file:
            data = json.load(json_file)
            paper_id = data['paper_id']
            
            # text = [item['text'] for item in data['body_text']]
            for item in data['body_text']:
                text = (item['text'])
                
                d_acc[paper_id] = text
                
    return d_acc
# json files'path from each folder

# path if needed to check just one article.
biomed_path = "../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/"  # bio and med archive
commu_path = "../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/"
noncom_path = "../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/"
pmc_path = "../input/CORD-19-research-challenge/pmc_custom_license/pmc_custom_license/"

# path if needed to check over all the folder.
biomed_fo = "../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/*.json"  # bio and med archive
commu_fo = "../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/*.json"
noncom_fo = "../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/*.json"
pmc_fo = "../input/CORD-19-research-challenge/pmc_custom_license/pmc_custom_license/*.json"
metadata = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")
metadata.head(5)
def met_xiv(metadata, sha):
    
    sha = str(sha)
        
    for i, tracksha in enumerate(metadata['sha']):
        if tracksha == sha:
            print("Title\n{} \n\nAuthors\n{} \n\nSource: {} \n\nPaper ID: {}\n\ndoi: {} \n\npmcid: {} - pubmed_id: {} \n\nJournal: {}\n\n-linked to-\n\nMicrosoft Academic Paper ID: {} \n\nWHO #Covidence: {}".format(
                                                                                                                                                                                                 metadata['title'][i],
                                                                                                                                                                                                 metadata['authors'][i],
                                                                                                                                                                                                 metadata['source_x'][i],
                                                                                                                                                                                                 metadata['sha'][i],
                                                                                                                                                                                                 metadata['doi'][i],
                                                                                                                                                                                                 metadata['pmcid'][i],
                                                                                                                                                                                                 metadata['pubmed_id'][i],
                                                                                                                                                                                                 
                                                                                                                                                                                                 metadata['journal'][i],
                                                                                                                                                                                                 
                                                                                                                                                                                                 metadata['Microsoft Academic Paper ID'][i],
                                                                                                                                                                                                 metadata['WHO #Covidence'][i]))
# met_xiv(metadata, '0015023cc06b5362d332b3baf348d11567ca2fbb')
# import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_sci_md")

def match_it(theme, xiv):
    
    phMatch = PhraseMatcher(nlp.vocab)
    
    article = data_access(xiv)
    
    
    p = key_per_theme(fact_name, descp)[theme]
    if len(p)==0:
        print('no patterns')
        
    patterns = [nlp(i) for i in p]

    phMatch.add(theme, None, *patterns)

    for num_id in article:
        paper_id = num_id
        
        doc = nlp(article[num_id])
        
        mat = phMatch(doc)
        # print(mat)
        
        for match_id, start, end in mat:
            string_id = nlp.vocab.strings[match_id]
            if len(string_id) == 0:
                print("No Result")
            else:
                span = doc[start:end]
                spant = doc[(start) : (end+20)]

                print("\nTHEME: \033[34m{}\033[00m - KEYWORDS: \033[32m{}\033[00m\n\nQUOTE: \033[0;37;40m{}\033[00m\n\nPAPER_ID:{}".format(string_id,
                                                                                                                     span.text, spant.text, paper_id))

            
                print()
match_it('Susceptibility', biomed_fo)
met_xiv(metadata,'564f8823050b52b5f5c36638ac1ae07557963f36')
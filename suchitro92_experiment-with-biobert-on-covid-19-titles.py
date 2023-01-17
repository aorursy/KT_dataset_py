import os, json
import numpy as np
import pandas as pd
# To call biobert_embedding, we need to change directory (there is a more elegant way, but sorry for this)

os.chdir('../input/biobertcustom/')
from biobert_embedding import BertSim

# Let us call the BioBERT Sentence Embedding (BertSim) here, and set it to predict mode

bs = BertSim()
bs.set_predict()

# Now let us change back to the data directory

os.chdir('../CORD-19-research-challenge/')
# A function to collect the titles from the given input path folder

def collectTitle(input_path): 
    '''
        Return all the article's title from the input_path folder
    '''
    all_title = []
    jsonlist     = os.listdir(input_path)
    for jsonname in jsonlist: 
        jsonfile = input_path + jsonname
        with open(jsonfile) as f:
            jsondata = json.load(f)
            all_title.append((jsondata['metadata']['title'], input_path.split('/')[-2] + '/' + jsonname))
    return all_title
# This may take some time, please be patient and do not close this notebook :(

alltitles = []

biorxiv_pdf = collectTitle('biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/')
commuse_pdf = collectTitle('comm_use_subset/comm_use_subset/pdf_json/')
commuse_pmc = collectTitle('comm_use_subset/comm_use_subset/pmc_json/')
noncommuse_pdf = collectTitle('noncomm_use_subset/noncomm_use_subset/pdf_json/')
noncommuse_pmc = collectTitle('noncomm_use_subset/noncomm_use_subset/pmc_json/')
customlics_pdf = collectTitle('custom_license/custom_license/pdf_json/')
customlics_pmc = collectTitle('custom_license/custom_license/pmc_json/')

alltitles.extend(biorxiv_pdf)
alltitles.extend(commuse_pdf)
alltitles.extend(commuse_pmc)
alltitles.extend(noncommuse_pdf)
alltitles.extend(noncommuse_pmc)
alltitles.extend(customlics_pdf)
alltitles.extend(customlics_pmc)

alltitles = pd.DataFrame(alltitles)
alltitles.columns = ['sentence', 'nameoffile']
alltitles = alltitles[['nameoffile', 'sentence']]
# clean_text function to remove certain words from the title

def clean_text(temp):
    temp = temp.lower()
    if any('title'in word for word in temp.split(' ')[:4]):
        temp=temp.replace('title: ','').replace('title page ','').replace('title (provisional) ','')
        temp=temp.replace('title 1 ','').replace('title 4 ','').replace('title page: 1 ', '')
        temp=temp.replace('• title ','').replace('subject areas title ','').replace('title -','')
        temp=temp.replace('watching brief title ','').replace('title ','').replace('title','')
    return temp

# Return the cleaned titles

alltitles['sentence'] = alltitles['sentence'].apply(lambda x: clean_text(x))
alltitles = alltitles.loc[(alltitles['sentence'] != '') & ~pd.isna(alltitles['sentence'])]
alltitles = alltitles.reset_index(drop = True)
# Get all the title's embeddings

title_embeddings =  bs.bert_sentences_embedding(list(alltitles['sentence'].values))
alltitles['sentence_embedding'] = title_embeddings
def cos_sim(a,b):
    '''
        Return the cosine similarity between vector a and vector b
    '''
    if a is not None and b is not None:
        return  np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))
    else:
        return 0
# Next let us prepare the queries embedding

query0 = "are there geographic variations in the rate of covid-19 spread?"
query1 = "are there geographic variations in the mortality rate of covid-19?"
query2 = "is there any evidence to suggest geographic based virus mutation?"
q_e0 = bs.bert_sentences_embedding([query0])[0]
q_e1 = bs.bert_sentences_embedding([query1])[0]
q_e2 = bs.bert_sentences_embedding([query2])[0]
# Finally, let us check their similarities

alltitles['similarity_0'] = alltitles.apply(lambda x: cos_sim(x['sentence_embedding'],q_e0),axis=1)
alltitles['similarity_1'] = alltitles.apply(lambda x: cos_sim(x['sentence_embedding'],q_e1),axis=1)
alltitles['similarity_2'] = alltitles.apply(lambda x: cos_sim(x['sentence_embedding'],q_e2),axis=1)
list(alltitles.sort_values(['similarity_0'], ascending = False)['sentence'].reset_index(drop = True).loc[0:49].values)
list(alltitles.sort_values(['similarity_1'], ascending = False)['sentence'].reset_index(drop = True).loc[0:49].values)
list(alltitles.sort_values(['similarity_2'], ascending = False)['sentence'].reset_index(drop = True).loc[0:49].values)
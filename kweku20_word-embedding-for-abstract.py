import json
import re
import os
#Getting List of json files
biorxiv_medrxiv_pdfs = os.listdir('../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json')
print(len(biorxiv_medrxiv_pdfs))
all_abstract = []
for pdf in biorxiv_medrxiv_pdfs:

    with open('../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'+pdf, 'r') as json_file:
            data=json_file.read()

    dataset = json.loads(data)

    abstracts = dataset['abstract']
    for abstract in abstracts:
        all_abstract.append(abstract['text'].split())

print(all_abstract[10])
from gensim.models import Word2Vec

word2vec = Word2Vec(all_abstract,size=100, window=5, min_count=5, workers=4, sg=0)
v1 = word2vec.wv['COVID-19']
print(v1)
sim_words = word2vec.wv.most_similar('COVID-19')
print(sim_words)
for sw in sim_words:
    print(sw)
from gensim.models import FastText
model_ted = FastText(all_abstract, size=100, window=5, min_count=5, workers=4,sg=1)
v1 = model_ted.wv['COVID-19']
print(v1)
ft_sim_words= model_ted.wv.most_similar("COVID-19")
print(ft_sim_words)
for sw in ft_sim_words:
    print(sw)
all_abstract = []
biorxiv_medrxiv_pdfs = os.listdir('../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json')
# print(len(biorxiv_medrxiv_pdfs))

for pdf in biorxiv_medrxiv_pdfs:

    with open('../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'+pdf, 'r') as json_file:
            data=json_file.read()

    dataset = json.loads(data)

    abstracts = dataset['abstract']
    for abstract in abstracts:
        all_abstract.append(abstract['text'].split())
comm_use_subset_pdfs = os.listdir('../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json')
print(len(comm_use_subset_pdfs))

for pdf in comm_use_subset_pdfs:

    with open('../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json/'+pdf, 'r') as json_file:
            data=json_file.read()

    dataset = json.loads(data)

    abstracts = dataset['abstract']
    for abstract in abstracts:
        all_abstract.append(abstract['text'].split())

custom_license_pdfs = os.listdir('../input/CORD-19-research-challenge/custom_license/custom_license/pdf_json')

for pdf in custom_license_pdfs:

    with open('../input/CORD-19-research-challenge/custom_license/custom_license/pdf_json/'+pdf, 'r') as json_file:
            data=json_file.read()

    dataset = json.loads(data)

    abstracts = dataset['abstract']
    for abstract in abstracts:
        all_abstract.append(abstract['text'].split())
noncomm_use_subset_pdfs = os.listdir('../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json')

for pdf in noncomm_use_subset_pdfs:

    with open('../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json/'+pdf, 'r') as json_file:
            data=json_file.read()

    dataset = json.loads(data)

    abstracts = dataset['abstract']
    for abstract in abstracts:
        all_abstract.append(abstract['text'].split())
print(len(all_abstract))
print(all_abstract[20])
from gensim.models import Word2Vec

word2vec = Word2Vec(all_abstract,size=100, window=5, min_count=5, workers=4, sg=0)
v1 = word2vec.wv['COVID-19']
print(v1)
sim_words = word2vec.wv.most_similar('COVID-19')
print(sim_words)
for sw in sim_words:
    print(sw)
from gensim.models import FastText
model_ted = FastText(all_abstract, size=100, window=5, min_count=5, workers=4,sg=1)
v1 = model_ted.wv['COVID-19']
print(v1)
ft_sim_words= model_ted.wv.most_similar("COVID-19")
print(ft_sim_words)
for sw in ft_sim_words:
    print(sw)

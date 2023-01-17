!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz

!pip install MulticoreTSNE



import scispacy

import en_ner_bc5cdr_md



#Faster T-SNE implementation

from MulticoreTSNE import MulticoreTSNE as TSNE
import os

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm



from sklearn.feature_extraction.text import TfidfVectorizer



import numpy as np

import matplotlib.pyplot as plt



import spacy

from spacy import displacy

from collections import Counter

import pickle

from collections import defaultdict



import seaborn as sns # plotting



%matplotlib inline
metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")

metadata = metadata.fillna("")
nlp = en_ner_bc5cdr_md.load()
dc_vocab = []

d2c_pairs = []



for row in tqdm(metadata.iterrows()):

    doc = nlp(row[1]["abstract"])

    dc_vocab.extend(doc.ents)

                

    for sent in doc.sents:

        for i in sent.ents:

            if i.label_ == "DISEASE":

                for j in sent.ents:

                    if j.label_ == "CHEMICAL":

                        d2c_pairs.append(((i.text, i.label_), (j.text, j.label_)))
pickle.dump([a.as_doc() for a in dc_vocab], open("docs.pk", "wb"))

pickle.dump(d2c_pairs, open("d2c_pairs.pk", "wb"))
# UNPACKING

# dc_vocab = pickle.load(open("/kaggle/input/cord-19/docs.pk", "rb"))

# dc_vocab = [a.ents[0] for a in dc_vocab]



# d2c_pairs = pickle.load(open("/kaggle/input/d2c-pairs/d2c_pairs.pk", "rb"))
diseases = [a for a in dc_vocab if a.label_ == "DISEASE"]

common_diseases = [a[0] for a in Counter([a.text for a in diseases]).most_common()[:300]]

print(common_diseases)



disease_set = set()

mini_batch = []



for a in dc_vocab:

    if a.text in common_diseases and a.text not in disease_set:

        mini_batch.append(a)

        disease_set.add(a.text)
nomenclature = pd.read_csv("/kaggle/input/drug-namestems/drug_nomenclature (1).csv")

nomenclature.drop(columns=['web-scraper-order', 'web-scraper-start-url'])

drug_namestems = list(nomenclature["affix_suffix"])
#Helper Function. It matches a string with a stem. 



def dstem_match(sstr, stem):

    pruned = stem.replace("-", "")

    stem_len = len(pruned)

    

    if stem[0] == "-" and stem[-1] != "-":

        if sstr[-stem_len:] == pruned:

            return True

    elif stem[-1] == "-" and stem[0] != "-":

        if sstr[:stem_len] == pruned:

            return True

    else:

        if pruned in sstr:

            return True

    

    return False
key_list = [a[0][0][0] for a in list(Counter([a for a in d2c_pairs if a[1][0] == "SARS-CoV-2"]).most_common())]
relevant_pairs = [a for a in d2c_pairs if a[0][0] in key_list] 
filtered = []



for pair in relevant_pairs:

    if dstem_match(pair[1][0], "-vir"):

        filtered.append(pair)

            

filtered
items = [(i[0][0], i[1][0]) for i in filtered]

dc_df = pd.DataFrame(items, columns =['disease', 'chemical']) 

cmatrix = pd.crosstab(dc_df.disease, dc_df.chemical)



cmatrix
sns.heatmap(cmatrix, cmap=plt.cm.cubehelix_r)
disease_groups = list(set([a[0][0] for a in filtered]))

len(disease_groups)
added = set()

mini_batch = []



for a in dc_vocab:

    if a.text in disease_groups and a.text not in added:

        added.add(a.text)

        mini_batch.append(a)

        

annotations = [a.text for a in mini_batch]



X_embedded = TSNE(n_components=2, verbose=1).fit_transform(np.array([a.vector for a in mini_batch]))

X_embedded.shape



# kmeans = KMeans(n_clusters=10, random_state=5, verbose=1).fit(X_embedded)



plt.figure(figsize=(10,10))

cdict = {1: 'red', 2: 'blue', 3: 'green', 4: 'yellow', 5: 'purple'}

plt.scatter(X_embedded[:, 0], X_embedded[:, 1])



for i, txt in enumerate(annotations):

    plt.annotate(txt, (X_embedded[:, 0][i], X_embedded[:, 1][i]))



plt.show()
!nvidia-smi
!cat /usr/local/cuda/version.txt
## Passing Y as input while conda asks for confirmation, we use yes command

!yes Y | conda install faiss-gpu cudatoolkit=10.0 -c pytorch
# !wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2

# !tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2

# !cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/

# # !export LD_LIBRARY_PATH="/kaggle/working/lib/" 

# !cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/
!wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'

!cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/

# !export LD_LIBRARY_PATH="/kaggle/working/lib/" 

!cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/
!apt search openblas

!yes Y | apt install libopenblas-dev 

# !printf '%s\n' 0 | update-alternatives --config libblas.so.3 << 0

# !apt-get install libopenblas-dev 

!
import faiss

from tsnecuda import TSNE

import pandas as pd

import numpy as np

from  sklearn.manifold import TSNE as sktsne

import matplotlib.pyplot as plt

import seaborn as sns

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
doc2vec = Doc2Vec.load("../input/sumotickets/mturk_doc2vec.genism.model")
print(doc2vec.wv.most_similar('bookmark'))
# tsne_model = TSNE(n_components=2, perplexity=40.0, n_iter=2000).fit_transform(X)

doc_tags = doc2vec.docvecs.doctags.keys()

X = doc2vec[doc_tags]

tsne_model = TSNE(n_components=2, perplexity=40.0).fit_transform(X)
df = pd.DataFrame(tsne_model, index=doc_tags, columns=['x', 'y'])
# tsne_df = pd.DataFrame(tsne_model)

# # tsne_df = pd.concat([tsne_df,Y], axis=1)

df_tickets = pd.read_csv("../input/sumoticket/all_annotated_tickets_clean.csv")
# df_tickets.iloc[:, 13:274]

REGULAR_TAGS = "regular_tags"

MTURK_TAGS = "mturk_tags"



# df_tickets[MTURK_TAGS] = df_tickets.iloc[:, 13:274].astype(str).agg('|'.join, axis=1)

df_tickets[REGULAR_TAGS] = df_tickets.iloc[:, 3:13].astype(str).agg('|'.join, axis=1)



# len(df_tickets.columns)

# df_tickets[df_tickets["id"] == int("808689")].iloc[:, 3:13]
def plotScatter(tag):

    fig = plt.figure(figsize=(10,15))

    ax = fig.add_subplot(1, 1, 1)

    

    pos_found_x = []

    pos_found_y = []

    found_names = []



    pos_rest_x = []

    pos_rest_y = []

    

    for term_id, pos in df.iterrows():

#         term_row = df_tickets[df_tickets["id"] == int(term_id)]

        term_regular_tags = df_tickets[df_tickets["id"] == int(term_id)][REGULAR_TAGS].tolist()[0].split("|")

#         term_mturk_tags = term_row[MTURK_TAGS].split("|") #[t for t in term_row.iloc[:, 13:274].astype(str).agg("|".join) if t != "nan"]



        if tag in term_regular_tags:

            pos_found_x.append(pos['x'])

            pos_found_y.append(pos['y'])

        else:

            pos_rest_x.append(pos['x'])

            pos_rest_y.append(pos['y']) 



    ax.scatter(pos_rest_x, pos_rest_y, c='blue')       

    ax.scatter(pos_found_x, pos_found_y, c='red')

plotScatter("bookmarks")

# df_tickets.columns

# df_tickets[df_tickets["id"] == 808689][REGULAR_TAGS].tolist()[0].split("|")
# sns.FacetGrid(tsne_df, hue="label" , size=6).map(plt.scatter, 0, 1).add_legend()

# plt.show()
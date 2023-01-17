# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
total_files = 0
for dirname, _, filenames in os.walk('/kaggle/input'):
    total_files += len(filenames)
    # for filename in filenames:
    #    print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
print("COVID-19 Research Analysis - checksum, total files = " + str(total_files))
metadata = pd.read_csv('/kaggle/input/CORD-19-research-challenge/metadata.csv')
metadata.head()
len(metadata.journal.unique())
!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import spacy
import scispacy
from tqdm import tqdm
def cosine_similarity(u, v):
    """
    Coded this as part of deeplearning.ai sequence modeling course
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0

    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u, v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u * u))
    
    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v * v))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)
    
    return cosine_similarity
nlp_lg = spacy.load("en_core_sci_lg")
vector_dict_lg = {}
for sha, abstract in tqdm(metadata[["sha","abstract"]].values):
    if isinstance(abstract, str):
        vector_dict_lg[sha] = nlp_lg(abstract).vector
keys_lg = list(vector_dict_lg.keys())
values_lg = list(vector_dict_lg.values())
print("top 5 keys = {keys}, and values = {values}".format(keys = keys_lg[0:5], values = values_lg[0:5]))
valarray_lg = np.asarray(values_lg, dtype=np.float32)
cosine_sim_matrix_lg = cosine_similarity(valarray_lg, valarray_lg.T)
print(type(cosine_sim_matrix_lg))
# same SHA as before
input_sha = "aecbc613ebdab36753235197ffb4f35734b5ca63"
n_sim_articles = 5


sha_index_lg = keys_lg.index(input_sha)
sim_indexes_lg = np.argsort(cosine_sim_matrix_lg[sha_index_lg])[::-1][1:n_sim_articles+1]
sim_shas_lg = [keys_lg[i] for i in sim_indexes_lg]
meta_info_lg = metadata[metadata.sha.isin(sim_shas_lg)]
print("=====QUERY ABSTRACT=====")
print(metadata[metadata.sha == input_sha]["abstract"].values[0])
print(f"=====TOP {n_sim_articles} SIMILAR ABSTRACTS USING LARGE MODEL=====")
for abst in meta_info_lg.abstract.values:
    print(abst)
    print("=======")
n_return = 5
nl_query_statement = "Studies showing discrepancy between humoral and cellular immunity in genetically similar subjects may be significant in the pathogenesis of systemic lupus erythematosus (SLE)."
query_vector_lg = nlp_lg(nl_query_statement).vector
cosine_sim_matrix_query_lg = cosine_similarity(valarray_lg, query_vector_lg)
query_sim_indexes_lg = np.argsort(cosine_sim_matrix_query_lg.reshape(1,-1)[0])[::-1][:n_return]
query_shas_lg = [keys_lg[i] for i in query_sim_indexes_lg]
meta_info_query_lg = metadata[metadata.sha.isin(query_shas_lg)]
print("=====QUERY ABSTRACT=====" + nl_query_statement)
print(f"=====TOP {n_sim_articles} SIMILAR ABSTRACTS USING LARGE MODEL=====")
for abst in meta_info_query_lg.abstract.values:
    print(abst)
    print("=======")
n_return = 5
gn_query_statement = "Effectiveness of drugs being developed and tried to treat COVID-19 patients."
gn_query_vector = nlp_lg(gn_query_statement).vector
gn_cosine_sim_matrix_query = cosine_similarity(valarray_lg, gn_query_vector)
gn_query_sim_indexes = np.argsort(gn_cosine_sim_matrix_query.reshape(1,-1)[0])[::-1][:n_return]
gn_query_shas = [keys_lg[i] for i in gn_query_sim_indexes]
gn_meta_info_query = metadata[metadata.sha.isin(gn_query_shas)]
print("=====QUERY ABSTRACT=====" + gn_query_statement)
print(f"=====TOP {n_sim_articles} SIMILAR ABSTRACTS USING LARGE MODEL=====")
for sha, abst in zip(gn_query_shas, gn_meta_info_query.abstract.values):
    print("sha: " + sha)
    print("abstract:" + abst)
    print("=======")
from sklearn.manifold import TSNE
t_sne = TSNE(verbose=1, perplexity=5)
abstractsvec = t_sne.fit_transform(valarray_lg)
from sklearn.cluster import MiniBatchKMeans

k = 10
mini_batch_kmeans = MiniBatchKMeans(n_clusters=k)
cat_pred = mini_batch_kmeans.fit_predict(valarray_lg)
cat = cat_pred
from matplotlib import pyplot as plt
import seaborn as sns
import random 

# seaboarn settings
sns.set(color_codes=True)
sns.set(rc={'figure.figsize':(12,12), 'axes.facecolor':'0.25'})

colors = sns.hls_palette(10, l = .5, s = .75)
random.shuffle(colors)

# plot
sns.scatterplot(abstractsvec[:,0], abstractsvec[:,1], hue = cat, legend = 'full', palette = colors)
plt.title("Plot of the clusterings of the abstracts from COVID-19 challenge dataset (wordvec model = scispaCy large, Clustering = MiniBatchKMeans )")
plt.show()

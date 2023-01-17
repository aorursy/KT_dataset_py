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
! pip install setuptools-rust
! pip install sentence-transformers
!pip install nmslib
import nmslib
import numpy as np
import os
import scipy
import pandas as pd
from IPython.display import display, HTML
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer, AutoModel

DATA_PATH = "/kaggle/input/"
text_data = pd.read_csv(os.path.join(DATA_PATH,"uspstf-guidelines/uspstf-with-id.txt"), sep="\t", index_col="id")
text_data.head(10)
def load_model(model_name, **pooling_mode):
    """
    Loads HuggingFace model and a Pooling Model into a SentenceTransformer object. HuggingFace is an external 
    repository of Pretrained Bert Models and the Pooling Model makes the Embeddings the same shape

    Args:
        model_name (str): name of a model listed on hugging face at https://huggingface.co/models
        pooling_mode (dict): map of keyword tokens describing tokens in Pooling Model

    Returns:
        SentenceTransformer: instance of SentenceTransformer loaded with HuggingFace model and Pooling Model
    """
    transformer = models.Transformer(model_name)
    pooling_model = models.Pooling(transformer.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=pooling_mode.get("mean_tokens", True),
                                   pooling_mode_cls_token=pooling_mode.get("cls_token", False),
                                   pooling_mode_max_tokens=pooling_mode.get("max_token", False))

    model = SentenceTransformer(modules=[transformer, pooling_model])
    return model

def encode_text(model, text):
    """
    Encodes a list of strings using a SentenceTransformer Model

    Args:
        model (SentenceTransformer): Embedding model
        text (List[str]): text to be encoded
    Returns:
        List[np.array]: List of vectors representing the inputted text data
    """
    encodings = []
    for t in text:
        encodings.append(model.encode(t)[0])
    return encodings

def load_queries():
    """
    Loads a list of queries to test SentenceTransformer models tagged with the id of the guideline they could be
    associated with.

    Returns:
        Dataframe: queries tagged with the ids of the guidelines they could be associated with
    """
    queries = pd.read_csv("/kaggle/input/cpgsearchgolddata/clinical_search_test.txt", sep="\t")
    display(queries.head(10))
    return queries

def run_queries(queries, model, encodings):
    """
    Matches a set of queries with the encodings and tests if a guideline associated with a question is in the 
    top 1, 5 or 10 results from the output of the matching operation

    Args:
        queries (Dataframe): Dataframe contain queries and ids of matching guidelines
        model (SentenceTransformer): Embedding model
        encodings (List[str]): vectors of encoded text
    Returns:
        int, int, int, int: counts of the count of top_1, top_, top_10 and the number of queries tested
    """
    rows = []
    top_1 = 0
    top_5 = 0
    top_10 = 0
    for i, row in queries.iterrows():
        query_vector = model.encode(row["query"])

        labels = set([int(x.strip()) for x in row["correct_match"].strip("\"").split(",")])
        distances = scipy.spatial.distance.cdist(query_vector, encodings, "cosine")[0]

        results = list(zip(range(len(distances)), distances))
        results = [x[0] for x in sorted(results, key=lambda x: x[1])]
        if len(labels.intersection(set(results[:1]))) > 0:
            top_1 += 1
        if len(labels.intersection(set(results[:5]))) > 0:
            top_5 += 1
        if len(labels.intersection(set(results[:10]))) > 0:
            top_10 += 1
    query_num = len(queries)
    
    return top_1, top_5, top_10, query_num
        
bert_models = ['bert-base-nli-mean-tokens', "emilyalsentzer/Bio_ClinicalBERT","allenai/biomed_roberta_base",
               "monologg/biobert_v1.1_pubmed", "bert-base-uncased", "allenai/scibert_scivocab_uncased"]
queries = load_queries()

for model_name in bert_models:
    print(f"Model {model_name}")
    if model_name != 'bert-base-nli-mean-tokens':
        model = load_model(model_name)
    else:
        model = SentenceTransformer('bert-base-nli-mean-tokens')
    encodings = encode_text(model, list(text_data["guideline"]))
    top_1, top_5, top_10, query_num = run_queries(queries, model, encodings)
    print("Top 1 result match: {} out of {} {:.2f}".format(top_1, query_num, top_1/query_num))
    print("Top 5 result match: {} out of {} {:.2f}".format(top_5, query_num, top_5/query_num))
    print("Top 10 result match: {} out of {} {:.2f}".format(top_10, query_num, top_10/query_num))
    print()

model = SentenceTransformer('bert-base-nli-mean-tokens')

transformer = models.Transformer("bert-base-uncased")
pooling_model = models.Pooling(transformer.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[transformer, pooling_model])

encodings = model.encode(text_data["guideline"])

display(HTML('<h1>Input</h1>'))
query = input()
print()
display(HTML('<h1>Results:</h1>'))
query_vector = model.encode([query])
distances = scipy.spatial.distance.cdist(query_vector, encodings, "cosine")[0]

results = list(zip(range(len(distances)), distances))
results = sorted(results, key=lambda x: x[1])[:10]
for index, _ in results:
    topic, description = text_data.iloc[index]["guideline"].split(" - ")
    display(HTML(f'<h3>{topic}</h3><p>{description}</p>'))
    print()
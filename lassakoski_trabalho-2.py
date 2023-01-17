import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import ward, fcluster, linkage
from scipy.spatial.distance import pdist
import nltk
import plotly.figure_factory as ff
from scipy import stats

#from google.colab import files
import io

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#abrir Tweets.csv diretamente de uma pasta no seu computador.
#Executar a celula e usar o "choose files" para selecionar o arquivo "Tweets.csv"

#uploaded = files.upload()
#tweets = pd.read_csv(io.BytesIO(uploaded['Tweets.csv']))

tweets = pd.read_csv('/kaggle/input/data-trab-2/Tweets.csv',error_bad_lines=False)
#visualizar tweets
tweets
#comments = tweets['text']
#comments

doc = tweets['text']
doc
# limpeza dos textos

wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # apenas letras minusculas, remover caracteres especiais e espaços em branco
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = doc.lower()
    doc = doc.strip()
    # extrair os tokens (palavras)
    tokens = wpt.tokenize(doc)
    # filtrar stopwords
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # recriar o documento com as palavras filtradas
    doc = ' '.join(filtered_tokens)
    return doc

normalize_doc = np.vectorize(normalize_document)

norm_doc = normalize_doc(doc)
norm_doc
#Importar modelo do USE diretamente do tensorflow hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

#função que recebe os tweets pré-processados e retorna os embeddings
def embed(input):
    return model(input)
# Embeddings
X = embed(norm_doc)
Y = pdist(X,'cosine')
Z = linkage(y = Y)
# Os clusters
F=fcluster(Z,t = 0.2)
F
# Utilizo a moda para observar o cluster mais frequente
stats.mode(F)
# O cluster mais frequente é a repetição da mesma frase united thank
df_texto = pd.DataFrame({'texto': norm_doc, 'cluster': F}, columns=['texto', 'cluster'])
filtro = df_texto.cluster == 2937
df_texto[filtro]
# Aqui vemos ele clusterizando prazes semelhantes
df_texto = pd.DataFrame({'texto': norm_doc, 'cluster': F}, columns=['texto', 'cluster'])
filtro = df_texto.cluster == 15
df_texto[filtro]
# o dendograma dividie os clusters em 3 grupos
ff.create_dendrogram(Z)

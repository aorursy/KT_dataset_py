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
        pass
        #print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
len(os.listdir("/kaggle/input/qa-testing-23-june"))
os.listdir('/kaggle/input/qa-testing-23-june')
import os
os.chdir("/kaggle/input/qa-testing-23-june/old_transformers")
!pip install -U ./transformers > /dev/null
os.chdir("/kaggle/input")
!pip install -q gensim==3.8.2 > /dev/null
!pip install sentence-transformers > /dev/null
import warnings
warnings.filterwarnings("ignore")
import os
import json
import pandas as pd 
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec
import numpy as np
import torch
from transformers import LongformerTokenizer, LongformerForQuestionAnswering, LongformerTokenizerFast
import matplotlib.pyplot as plt
list_of_docs = os.listdir("/kaggle/input/qa-testing-23-june/Data_For_Question_Answering_on_Regulatory_Documents-20200623T050102Z-001/Data_For_Question_Answering_on_Regulatory_Documents/Annotated_QAs")
#print(list_of_docs)
print(len(list_of_docs))
list_of_docs = ["/kaggle/input/qa-testing-23-june/Data_For_Question_Answering_on_Regulatory_Documents-20200623T050102Z-001/Data_For_Question_Answering_on_Regulatory_Documents/Annotated_QAs/{}".format(i) for i in list_of_docs if "train-v1.1.json" not in i and "dev-v1.1.json" not in i]
#print(list_of_docs)
print(len(list_of_docs))
docs = []

for i in list_of_docs:
  with open(i, 'r') as f:
    doc = json.load(f)
  docs += [i['context'] for i in doc['data'][0]['paragraphs']]

print(len(docs))
#docs[0]
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def clean_docs(x, bi = True):
  x = re.sub(r"n't", r" not", x.lower())
  x = re.sub(r"[^-0-9a-z ]", r" ", x)
  x = re.sub(r"[ ]+", r" ", x)
  x = word_tokenize(x)
  x = [i for i in x if len(i) > 1]
  x = [stemmer.stem(i) for i in x]
  if bi == True:
    return bigram[x]
  else:
    return x
print(clean_docs("Money laundering and terror financing and money launderer and terror financer", bi = False))
news_documents = ["/kaggle/input/qa-testing-23-june/drive-download-20200623T104925Z-001/{}".format(i) for i in os.listdir("/kaggle/input/qa-testing-23-june/drive-download-20200623T104925Z-001") if ".xlsx" in i and "news" in i]

df = pd.DataFrame()

for i in news_documents:
  df = pd.concat((df, pd.read_excel(i)), axis = 0)

docs_2 = df.Text.tolist()

print(len(docs_2))

docs_2 = list(set(docs_2))

docs_2 = [i for i in docs_2 if type(i) == type('s')]

docs_2 = [i for i in docs_2 if len(i) > 100]

docs_2 = [i for i in docs_2 if ("money" in i.lower() and "laundering" in i.lower()) or ("aml" in i.lower() and "ctf" in i.lower()) or ("pep" in i.lower() and "state" in i.lower() and "bank" in i.lower()) or ("state" in i.lower() and "bank" in i.lower()) or ("compliance" in i.lower())]

print(len(docs_2))
docs_for_w2v = docs + docs_2
docs_for_w2v_prime = [clean_docs(i, bi = False) for i in docs_for_w2v]
print(docs_for_w2v_prime[0:10])
print(len(docs_for_w2v_prime))
bigram = Phraser(Phrases(docs_for_w2v_prime, threshold = 64))
docs_for_w2v_prime_2 = [bigram[i] for i in docs_for_w2v_prime]
docs_for_w2v_prime_2[:2]
model_w2v = Word2Vec(docs_for_w2v_prime_2, size = 64)
model_w2v.most_similar(clean_docs('money-laundering'))
#bigram.save('/content/gdrive/My Drive/bigram_22_june.pkl')
#model_w2v.save('/content/gdrive/My Drive/w2v_22_june.w2v')
bigram = Phraser.load('/kaggle/input/qa-testing-23-june/bigram_22_june.pkl')
model_w2v = Word2Vec.load('/kaggle/input/qa-testing-23-june/w2v_22_june.w2v')
from gensim.summarization.bm25 import BM25
vocab = model_w2v.wv.vocab

from nltk.corpus import stopwords
nltk.download('stopwords')

stopset = stopwords.words()

def art_to_vect(x):
  x = clean_docs(x)
  x = list(set(x))
  x = (np.array([model_w2v[i] for i in x if i in vocab and i not in stopset]))
  return np.sum(x, axis = 0).flatten()/len(x)
from sentence_transformers import SentenceTransformer

model_bert = SentenceTransformer('bert-base-nli-mean-tokens')

vect_bert = model_bert.encode(docs)
cleaned_chatbot_docs = [clean_docs(i) for i in docs]
vect_docs = [art_to_vect(i) for i in docs]
docs_tuple = [(i, j, k, l) for i, j, k, l in zip(docs, cleaned_chatbot_docs, vect_docs, vect_bert)]
def corpus(x):
  for i in x:
    yield i
bm25 = BM25(corpus(cleaned_chatbot_docs))
def get_top_n(bm25, query, n=10):
    
    # score docs
    scores = np.array(bm25.get_scores(query))
    
    # get indices of top N scores
    idx = np.argpartition(scores, -n)[-n:]
    
    # sort top N scores and return their indices
    return idx[np.argsort(-scores[idx])]
test_query = clean_docs("How is an offshore bank defined")
top_idx_bm = get_top_n(bm25, test_query)
print(top_idx_bm)
from scipy.spatial.distance import cosine

def get_top_n_vect(vects, query, n = 10):
  x = art_to_vect(query)
  x = [(j, 1 - cosine(x, i)) for j, i in enumerate(vects)]
  x = sorted(x, key = lambda x: x[1], reverse = True)[:10]
  x = [i[0] for i in x]
  return x
test_query = "How is an offshore bank defined?"
top_idx_w2v = get_top_n_vect(vect_docs, test_query)
print(top_idx_w2v)
#print(cleaned_chatbot_docs[top_idx])
#print(docs_tuple[top_idx][0])
def get_top_n_vect_bert(vects, query, n = 10):
  x = model_bert.encode([query])
  x = [(j, 1 - cosine(x, i)) for j, i in enumerate(vects)]
  x = sorted(x, key = lambda x: x[1], reverse = True)[:10]
  x = [i[0] for i in x]
  return x
test_query = "How is an offshore bank defined?"
top_idx_bert = get_top_n_vect_bert(vect_bert, test_query)
print(top_idx_bert)
#print(cleaned_chatbot_docs[top_idx])
#print(docs_tuple[top_idx][0])
print(set(top_idx_bm).intersection(top_idx_bert))
print(set(top_idx_bm).intersection(top_idx_w2v))
print(set(top_idx_bert).intersection(top_idx_w2v))
print(set(top_idx_bm).intersection(top_idx_bert).intersection(top_idx_w2v))
tokenizer = LongformerTokenizerFast.from_pretrained('/kaggle/input/qa-testing-23-june/CHATQA_4-20200623T050106Z-001/CHATQA_4')
model = LongformerForQuestionAnswering.from_pretrained('/kaggle/input/qa-testing-23-june/CHATQA_4-20200623T050106Z-001/CHATQA_4')

def qa_long(question, text):
  try:
    encoding = tokenizer.encode_plus(question, text, return_tensors="pt")
    input_ids = encoding["input_ids"]

    attention_mask = encoding["attention_mask"]

    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
    #print(torch.max(start_scores), torch.max(end_scores))
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    
    return answer, start_scores, end_scores
  except Exception as e:
    pass
import time
from textwrap import wrap

def check(query):
    with torch.no_grad():
        test_query = clean_docs(query)
        top_idx = get_top_n(bm25, test_query)
        print(query)
        for i in top_idx[0:5]:
            #print(cleaned_chatbot_docs[i])
            #print(docs_tuple[i][0])
            try:
                ans = qa_long(query, docs_tuple[i][0])
                #print(ans[0])
                #title = ax.set_title("\n".join(wrap("Some really really long long long title I really really need - and just can't - just can't - make it any - simply any - shorter - at all.", 60)))
                a = len(ans[0]) // 200
                if len(ans[0]) < 200:
                    ans[0] = ans[0].center(200)
                fig, (ax1, ax2) = plt.subplots(1, 2)
                fig.suptitle("\n".join(wrap(ans[0], 200)), y = 1 + int(a)/10)
                ax1.plot(ans[1].detach().numpy().flatten())
                ax2.plot(ans[2].detach().numpy().flatten())
                #plt.draw()
            except:
                pass
    #plt.show()
check("How is an offshore bank defined?")
questions = []

for i in list_of_docs:
  with open(i, 'r') as f:
    doc = json.load(f)
  x =  [i['qas'] for i in doc['data'][0]['paragraphs']]
  for i in x:
    y = [j['question'] for j in i]
    questions += y

print(len(questions))
questions[0]
import random

to_ask = [questions[random.randint(0, len(questions) - 1)] for i in range(10)]
check(to_ask[0])
check(to_ask[1])
check(to_ask[2])
check(to_ask[3])
check(to_ask[4])
check(to_ask[5])
check(to_ask[6])
check(to_ask[7])
check(to_ask[8])
check(to_ask[9])

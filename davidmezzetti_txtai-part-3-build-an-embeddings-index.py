%%capture

!pip install git+https://github.com/neuml/txtai
!wget https://www.kaggleusercontent.com/kf/40510829/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FX7Ote_I-Y88MBPQHRIdUQ.tTr7P3B_eUL_yWN33Usz0Rk1KXtc4DjT_cdjkl5W4WbEcZ-0FJX2jSWTHYMVACtLYuMJJrf6eJN28OzWhDMnTysBu3wfDrd4ly5bu_wJKCnZajICQgQHs_b8hbRVMOzfdG6xEyl9CVYnZNU2cI3QuOshcWxoB0skdKD4d26O_Q4e_nrd8DqEixP47tI2Hu1F00w0vMykzgNwp7SwQ2Z9HoNCO8HtmcjEHq0A4lZ4303YkpjORtZQEO3S-j54fFlIAahT-9VvsFNofitK5VAlR0EyG9r3cOqh2LQDCL7kj5p3MxG8dvHmrTqggLVOwiuKHUIH8u59TemSMLsNRS29W-5fFlHfaItV4dEuiBxCIgQXHcKUDCDGEjeFcPgqpJnNHsnh0pebWDuRQR_fdQ-r8mWgN9qLnosrFBak9tM25G7gqxyUI90GMWAUyP4yj2EAEc8asX9rUsirC8QDHmrmOCUe0cmZvodRUi0ss7lTiLTwm55d9VPXjQn4jQ6tFs-dmjXEx0AwF2Mw1c1jhgzCXwgQj6ybUKemr_6wj1VFYj3VVvCXpk1nZObl-IB6-m7v5CIoXGLot_KFsVtyItRk-wX-B_L3W3aS9dOIfb7bX4s5_aNzXaDKvxrcafwlOQui.vS_FL4EArO8rkBo3xpDF2w/articles.sqlite
import os

import sqlite3

import tempfile



from txtai.tokenizer import Tokenizer

from txtai.vectors import WordVectors



print("Streaming tokens to temporary file")



# Stream tokens to temp working file

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as output:

  # Save file path

  tokens = output.name



  db = sqlite3.connect("articles.sqlite")

  cur = db.cursor()

  cur.execute("SELECT Text from sections")



  for row in cur:

    output.write(" ".join(row[0]) + "\n")



  # Free database resources

  db.close()



# Build word vectors model - 300 dimensions, 3 min occurrences

WordVectors.build(tokens, 300, 3, "cord19-300d")



# Remove temporary tokens file

os.remove(tokens)



# Show files

!ls -l
import sqlite3



import regex as re



from txtai.embeddings import Embeddings

from txtai.tokenizer import Tokenizer



def stream():

  # Connection to database file

  db = sqlite3.connect("articles.sqlite")

  cur = db.cursor()



  # Select tagged sentences without a NLP label. NLP labels are set for non-informative sentences.

  cur.execute("SELECT Id, Name, Text FROM sections WHERE (labels is null or labels NOT IN ('FRAGMENT', 'QUESTION')) AND tags is not null")



  count = 0

  for row in cur:

    # Unpack row

    uid, name, text = row



    # Only process certain document sections

    if not name or not re.search(r"background|(?<!.*?results.*?)discussion|introduction|reference", name.lower()):

      # Tokenize text

      tokens = Tokenizer.tokenize(text)



      document = (uid, tokens, None)



      count += 1

      if count % 1000 == 0:

        print("Streamed %d documents" % (count), end="\r")



      # Skip documents with no tokens parsed

      if tokens:

        yield document



  print("Iterated over %d total rows" % (count))



  # Free database resources

  db.close()



# BM25 + fastText vectors

embeddings = Embeddings({"path": "cord19-300d.magnitude",

                         "scoring": "bm25",

                         "pca": 3})



# Build scoring index if scoring method provided

if embeddings.config.get("scoring"):

  embeddings.score(stream())



# Build embeddings index

embeddings.index(stream())

import pandas as pd



from IPython.display import display, HTML



pd.set_option("display.max_colwidth", None)



db = sqlite3.connect("articles.sqlite")

cur = db.cursor()



results = []

for uid, score in embeddings.search("risk factors", 5):

  cur.execute("SELECT article, text FROM sections WHERE id = ?", [uid])

  uid, text = cur.fetchone()



  cur.execute("SELECT Title, Published, Reference from articles where id = ?", [uid])

  results.append(cur.fetchone() + (text,))



# Free database resources

db.close()



df = pd.DataFrame(results, columns=["Title", "Published", "Reference", "Match"])



display(HTML(df.to_html(index=False)))
%%capture

from txtai.extractor import Extractor



# Create extractor instance using qa model designed for the CORD-19 dataset

extractor = Extractor(embeddings, "NeuML/bert-small-cord19qa")
db = sqlite3.connect("articles.sqlite")

cur = db.cursor()



results = []

for uid, score in embeddings.search("risk factors", 5):

  cur.execute("SELECT article, text FROM sections WHERE id = ?", [uid])

  uid, text = cur.fetchone()



  # Get list of document text sections to use for the context

  cur.execute("SELECT Id, Name, Text FROM sections WHERE (labels is null or labels NOT IN ('FRAGMENT', 'QUESTION')) AND article = ?", [uid])

  sections = []

  for sid, name, txt in cur.fetchall():

    if not name or not re.search(r"background|(?<!.*?results.*?)discussion|introduction|reference", name.lower()):

      sections.append((sid, txt))



  cur.execute("SELECT Title, Published, Reference from articles where id = ?", [uid])

  article = cur.fetchone()



  # Use QA extractor to derive additional columns

  answers = extractor(sections, [("Risk Factors", "risk factors", "What risk factors?", False),

                                 ("Locations", "hospital country", "What locations?", False)])



  results.append(article + (text,) + tuple([answer[1] for answer in answers]))



# Free database resources

db.close()



df = pd.DataFrame(results, columns=["Title", "Published", "Reference", "Match", "Risk Factors", "Locations"])

display(HTML(df.to_html(index=False)))
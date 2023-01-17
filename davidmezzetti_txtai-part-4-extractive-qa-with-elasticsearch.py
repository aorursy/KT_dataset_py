%%capture



# Install txtai and elasticsearch python client

!pip install git+https://github.com/neuml/txtai elasticsearch



# Download and extract elasticsearch

!wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.8.1-linux-x86_64.tar.gz

!tar -xzf elasticsearch-7.8.1-linux-x86_64.tar.gz

!chown -R daemon:daemon elasticsearch-7.8.1
import os

from subprocess import Popen, PIPE, STDOUT



# Start and wait for server

server = Popen(['elasticsearch-7.8.1/bin/elasticsearch'], stdout=PIPE, stderr=STDOUT, preexec_fn=lambda: os.setuid(1))

!sleep 30
!wget https://www.kaggleusercontent.com/kf/40510829/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..FX7Ote_I-Y88MBPQHRIdUQ.tTr7P3B_eUL_yWN33Usz0Rk1KXtc4DjT_cdjkl5W4WbEcZ-0FJX2jSWTHYMVACtLYuMJJrf6eJN28OzWhDMnTysBu3wfDrd4ly5bu_wJKCnZajICQgQHs_b8hbRVMOzfdG6xEyl9CVYnZNU2cI3QuOshcWxoB0skdKD4d26O_Q4e_nrd8DqEixP47tI2Hu1F00w0vMykzgNwp7SwQ2Z9HoNCO8HtmcjEHq0A4lZ4303YkpjORtZQEO3S-j54fFlIAahT-9VvsFNofitK5VAlR0EyG9r3cOqh2LQDCL7kj5p3MxG8dvHmrTqggLVOwiuKHUIH8u59TemSMLsNRS29W-5fFlHfaItV4dEuiBxCIgQXHcKUDCDGEjeFcPgqpJnNHsnh0pebWDuRQR_fdQ-r8mWgN9qLnosrFBak9tM25G7gqxyUI90GMWAUyP4yj2EAEc8asX9rUsirC8QDHmrmOCUe0cmZvodRUi0ss7lTiLTwm55d9VPXjQn4jQ6tFs-dmjXEx0AwF2Mw1c1jhgzCXwgQj6ybUKemr_6wj1VFYj3VVvCXpk1nZObl-IB6-m7v5CIoXGLot_KFsVtyItRk-wX-B_L3W3aS9dOIfb7bX4s5_aNzXaDKvxrcafwlOQui.vS_FL4EArO8rkBo3xpDF2w/articles.sqlite
import sqlite3



import regex as re



from elasticsearch import Elasticsearch, helpers



# Connect to ES instance

es = Elasticsearch(hosts=["http://localhost:9200"], timeout=60, retry_on_timeout=True)



# Connection to database file

db = sqlite3.connect("articles.sqlite")

cur = db.cursor()



# Elasticsearch bulk buffer

buffer = []

rows = 0



# Select tagged sentences without a NLP label. NLP labels are set for non-informative sentences.

cur.execute("SELECT s.Id, Article, Title, Published, Reference, Name, Text FROM sections s JOIN articles a on s.article=a.id WHERE (s.labels is null or s.labels NOT IN ('FRAGMENT', 'QUESTION')) AND s.tags is not null")

for row in cur:

  # Build dict of name-value pairs for fields

  article = dict(zip(("id", "article", "title", "published", "reference", "name", "text"), row))

  name = article["name"]



  # Only process certain document sections

  if not name or not re.search(r"background|(?<!.*?results.*?)discussion|introduction|reference", name.lower()):

    # Bulk action fields

    article["_id"] = article["id"]

    article["_index"] = "articles"



    # Buffer article

    buffer.append(article)



    # Increment number of articles processed

    rows += 1



    # Bulk load every 1000 records

    if rows % 1000 == 0:

      helpers.bulk(es, buffer)

      buffer = []



      print("Inserted {} articles".format(rows), end="\r")



if buffer:

  helpers.bulk(es, buffer)



print("Total articles inserted: {}".format(rows))

import pandas as pd



from IPython.display import display, HTML



pd.set_option("display.max_colwidth", None)



query = {

    "_source": ["article", "title", "published", "reference", "text"],

    "size": 5,

    "query": {

        "query_string": {"query": "risk factors"}

    }

}



results = []

for result in es.search(index="articles", body=query)["hits"]["hits"]:

  source = result["_source"]

  results.append((source["title"], source["published"], source["reference"], source["text"]))



df = pd.DataFrame(results, columns=["Title", "Published", "Reference", "Match"])



display(HTML(df.to_html(index=False)))
%%capture

from txtai.embeddings import Embeddings

from txtai.extractor import Extractor



# Create embeddings model, backed by sentence-transformers & transformers

embeddings = Embeddings({"method": "transformers", "path": "sentence-transformers/bert-base-nli-stsb-mean-tokens"})



# Create extractor instance using qa model designed for the CORD-19 dataset

extractor = Extractor(embeddings, "NeuML/bert-small-cord19qa")
document = {

    "_source": ["id", "name", "text"],

    "size": 1000,

    "query": {

        "term": {"article": None}

    },

    "sort" : ["id"]

}



def sections(article):

  rows = []



  search = document.copy()

  search["query"]["term"]["article"] = article



  for result in es.search(index="articles", body=search)["hits"]["hits"]:

    source = result["_source"]

    sid, name, text = source["id"], source["name"], source["text"]



    if not name or not re.search(r"background|(?<!.*?results.*?)discussion|introduction|reference", name.lower()):

      rows.append((sid, text))

  

  return rows



results = []

for result in es.search(index="articles", body=query)["hits"]["hits"]:

  source = result["_source"]



  # Use QA extractor to derive additional columns

  answers = extractor(sections(source["article"]), [("Risk factors", "risk factor", "What are names of risk factors?", False),

                                                    ("Locations", "city country state", "What are names of locations?", False)])



  results.append((source["title"], source["published"], source["reference"], source["text"]) + tuple([answer[1] for answer in answers]))



df = pd.DataFrame(results, columns=["Title", "Published", "Reference", "Match", "Risk Factors", "Locations"])



display(HTML(df.to_html(index=False)))
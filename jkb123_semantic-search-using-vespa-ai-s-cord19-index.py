!pip install transformers

!pip install sentence_transformers

import os

PATH = "/kaggle/working/model"

os.mkdir(PATH)
from transformers import AutoTokenizer

from transformers import AutoModelWithLMHead

from sentence_transformers import SentenceTransformer 

from sentence_transformers import models
tokenizer = AutoTokenizer.from_pretrained("gsarti/scibert-nli")

model = AutoModelWithLMHead.from_pretrained("gsarti/scibert-nli")

model.save_pretrained(PATH)

tokenizer.save_pretrained(PATH)

embedding = models.BERT(PATH,max_seq_length=128,do_lower_case=True)

pooling_model = models.Pooling(embedding.get_word_embedding_dimension(),

                               pooling_mode_mean_tokens=True,

                               pooling_mode_cls_token=False,

                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[embedding, pooling_model])

model.save(PATH)
encoder = SentenceTransformer(PATH)
embedding = encoder.encode(['Temperature significant change COVID-19 Transmission in 429 cities'])[0]
import requests 

vespa_body_ann = {

    'yql': 'select * from sources * where source contains "medrxiv" and ([{"targetNumHits":%i}]nearestNeighbor(title_embedding,vector));' % (5),

    'hits': 5,

    'ranking.features.query(vector)': embedding.tolist(),

    'ranking.profile': 'semantic-search-title',

    'summary': 'short',

    'timeout': '5s',

    'ranking.softtimeout.enable': 'false' 

}

response = requests.post('https://api.cord19.vespa.ai/search/', json=vespa_body_ann) 

result = response.json()

for hit in result['root']['children']:

    relevance = hit['relevance']

    id = hit['fields']['id']

    title = hit['fields'].get('title')

    print('Semantic Similarity=%f, title = %s' % (relevance, title))

import pprint

from IPython.display import Image

from ipywidgets import IntProgress

from IPython.display import display
data = {

    'institutions': [

        {

            'id': 1,

            'name': 'Austrian Centre for Digital Humanities',

            'acronym': 'ACDH',

            'type': 'academic institution',

            'uri': 'https://www.wikidata.org/wiki/Q30268470'

        },

        {

            'id': 2,

            'name': 'Österreichische Akademie der Wissenschaften',

            'acronym': 'ÖAW',

            'type': 'academic institution',

            'uri': 'https://www.wikidata.org/wiki/Q299015'

        }

    ],

    'relations': {

        'institution-institution': [

            {

                'id': 1,

                'type': 'parent of',

                'institutionA': 2,

                'institutionB': 1

            }

        ],

        'institution-place': [

            {

                'id': 1,

                'type': 'located at',

                'institution': 1,

                'place': 'https://www.wikidata.org/wiki/Q1741'

            },

            {

                'id': 2,

                'type': 'located at',

                'institution': 1,

                'place': 'https://www.wikidata.org/wiki/Q1741'

            }

        ]

    }

}
pp = pprint.PrettyPrinter(indent=4)

pp.pprint(data)
data['institutions'][0]['name']
from lxml import etree
file_path = '../input/oebl-test-xml/Defregger_Franz_1835_1921.xml'

with open(file_path) as file:

    et = etree.parse(file)

    print(et)
ns_oebl = {'xmlns': 'http://www.biographien.ac.at'}

with open(file_path) as file:

    et = etree.parse(file)

    name = et.xpath('.//xmlns:Hauptbezeichnung/text()', namespaces=ns_oebl)

    haupttext = et.xpath('.//xmlns:Haupttext[1]/text()', namespaces=ns_oebl)[0]

    print(name)

    print(haupttext)
import re
for year in re.finditer(r'\d{2,4}', haupttext):

    print(year.group(0))
for year in re.finditer(r'[\d\–]{2,7}', haupttext):

    print(year.group(0))
year_place = []

for year in re.finditer(r'([\d\–]{2,7}).*?\snach\s.*?([A-Z]\w+)', haupttext):

    print('year: {} / place: {}'.format(year.group(1), year.group(2)))

    year_place.append((year.group(1), year.group(2)))
sentences = []

for idx, sent in enumerate(re.split(r'[\.!?;]{1}', haupttext)):

    print('sentence #{}: {}'.format(idx, sent))

    sentences.append(sent)
Image("../input/neuronalnetwork/neuronal_net_v3_step2.png")
Image("../input/neuronalnetwork/neuronal_net_v3_step3.png")
!pip install spacy requests
import spacy
!python -m spacy download de
nlp = spacy.load('de')
txt = "Wien ist eine schöne Stadt in Österreich. Die Universität Wien liegt am Schottenring. Heinz Engel ist Rektor der Universität."
doc = nlp(txt)
for e in doc.ents:

    print(e.text, e.label_)
from spacy import displacy
displacy.render(doc, style="ent")
for t in doc:

    print(t.text, t.pos_)
displacy.render(doc, style="dep")
import requests

import spacy

import pandas as pd
url = 'https://apis.acdh.oeaw.ac.at/apis/api/entities/person/'

profession = 'Maler'

headers = {'accept': 'application/json'}

params = {'profession__name__icontains': profession, 'collection': '15'}

res = requests.get(url, headers=headers, params=params)
print(res.status_code)
res_json = res.json()

res_json['results'][:10]
f = IntProgress(min=0, max=15)

display(f)

res_df = []

prof_lst = dict()

for r in res_json['results'][:15]:

    f.value += 1

    p1 = {

        'person ID': r['id'],

        'name': r['name'],

        'first_name': r['first_name'],

        'birth_date': r['start_date'],

        'death_date': r['end_date'],

        'gender': r['gender'],

        'profession': [],

        'uris': [],

        'short info': '',

        'biography': '',

        'biography id': ''

    }

    for prof in r['profession']:

        if prof not in prof_lst:

            prof_res = requests.get(prof).json()

            prof_name = prof_res['name']

            prof_lst[prof] = prof_name

        else:

            prof_name = prof_lst[prof]

        p1['profession'].append(prof_name)

    for txt in r['text']:

        t1 = requests.get(txt).json()

        if '/131/' in t1['kind']:

            p1['short info'] = t1['text']

        elif '/130/' in t1['kind']:

            p1['biography'] = t1['text']

            p1['biography id'] = t1['id']

    params2 = {'entity': r['id']}

    uris = requests.get('https://apis.acdh.oeaw.ac.at/apis/api/metainfo/uri/', params=params2).json()

    p1['uris'] = ', '.join([x['uri'] for x in uris['results']])

    p1['profession'] = ', '.join(p1['profession'])

    res_df.append(p1)
res_df
maler_df = pd.DataFrame(res_df)
maler_df
nlp = spacy.load('de')
test_txt = maler_df.at[2,'biography']
test_txt
doc = nlp(test_txt)
displacy.render(doc, style='ent')
from spacy.symbols import ORTH

lst_abbrev = ['Wr.', 'Akad.', 'Stud.', 'hist.', 'lyr.', 'venezian.']

for abbrev in lst_abbrev:

    special_case = [{ORTH: abbrev}]

    nlp.tokenizer.add_special_case(abbrev, special_case)
doc = nlp(test_txt)
displacy.render(doc, style="ent")
train_data = [

    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),

    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),

]
import json

with open('../input/apis-test-data/train_data_test.json', 'r') as inp:

    TRAIN_DATA = json.load(inp)
import random

from pathlib import Path

import spacy

from spacy.util import minibatch, compounding

def train_ner_spacy(model=None, output_dir=None, n_iter=100):

    """Load the model, set up the pipeline and train the entity recognizer."""

    if model is not None:

        nlp = spacy.load(model)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")



    # create the built-in pipeline components and add them to the pipeline

    # nlp.create_pipe works for built-ins that are registered with spaCy

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe("ner")



    # add labels

    for _, annotations in TRAIN_DATA:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])



    # get names of other pipes to disable them during training

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    with nlp.disable_pipes(*other_pipes):  # only train NER

        # reset and initialize the weights randomly – but only if we're

        # training a new model

        if model is None:

            nlp.begin_training()

        for itn in range(n_iter):

            random.shuffle(TRAIN_DATA)

            losses = {}

            # batch up the examples using spaCy's minibatch

            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(

                    texts,  # batch of texts

                    annotations,  # batch of annotations

                    drop=0.5,  # dropout - make it harder to memorise data

                    losses=losses,

                )

            print("Losses", losses)



    # test the trained model

    for text, _ in TRAIN_DATA:

        doc = nlp(text)

        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])



    # save model to output directory

    if output_dir is not None:

        output_dir = Path(output_dir)

        if not output_dir.exists():

            output_dir.mkdir()

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)



        # test the saved model

        print("Loading from", output_dir)

        nlp2 = spacy.load(output_dir)

        for text, _ in TRAIN_DATA:

            doc = nlp2(text)

            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    return nlp2
nlp2 = train_ner_spacy(model='de', output_dir='test_model_apis', n_iter=10)
doc = nlp2(test_txt)
displacy.render(doc, style="ent")
url = 'http://enrich.acdh.oeaw.ac.at/entityhub/site/geoNames_S_P_A/find'

ldpath = "long = <http://www.w3.org/2003/01/geo/wgs84_pos#long>;\n"

ldpath += "lat = <http://www.w3.org/2003/01/geo/wgs84_pos#lat>;\n"

ldpath += "featureCode = <http://www.geonames.org/ontology#featureCode>;"
params = {'name': 'Wien', 'limit': 20, 'ldpath': ldpath}

headers = {'Content-Type': 'application/json'}

res = requests.get(url, params=params, headers=headers)

res.json()
for ent in doc.ents:

    if ent.label_ == 'LOC':

        params['name'] = ent.text

        res = requests.get(url, params=params, headers=headers).json()

        print(f"string: {ent.text} / results: {', '.join(x['id'] for x in res['results'])}")
import json
for ent in doc.ents:

    if ent.label_ == 'PER':

        query_lst = ent.text.split(' ')

        query_lst = [x.strip() for x in query_lst]

        query = ', '.join(reversed(query_lst))

        print(query)

        url='http://enrich.acdh.oeaw.ac.at/entityhub/site/gndPersons/find'

        params['name'] = query

        res = requests.get(url, params=params, headers=headers)

        print(res.status_code)

        if res.status_code == 200:

            res = res.json()

            print(f"string: {ent.text} / results: {', '.join(x['id'] for x in res['results'])}")
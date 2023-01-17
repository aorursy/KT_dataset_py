import requests

import spacy

from spacy import displacy

!python -m spacy download de
train_data = [

    ("Who is Shaka Khan?", {"entities": [(7, 17, "PER")]}),

    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),

]
import json

with open('/kaggle/input/apis-test-data/train_data_test.json', 'r') as inp:

    TRAIN_DATA = json.load(inp)
len(TRAIN_DATA)
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
test_txt = "Als Sohn eines Eisenbahn-Obering. verlebte er seine Jugend in Rekawinkel und in Wien, wandte sich dann an den Akad. in München (1873/74, O. Seitz) und Wien (1875–77, A. Feuerbach) dem Stud. der Malerei zu. Studienreisen führten ihn nach Frankreich, England, in den Orient und nach Amerika, unterbrochen durch längere Aufenthalte in München (1884–88), Dachau, Paris und in der Bretagne. 1904–07 Ausstattungschef des Wr. Burgtheaters und 1909/10 der Wr. Hofoper. Anfangs widmete sich G. im Anschluß an Feuerbach dem Figurenbild mit hist., oft auch oriental. Gegenstand, das er mit lyr. Stimmungen und venezian. Kolorit behandelte, sowie dem Porträt, dann ging er in Landschaft und Porträt zur Helldunkelmalerei über."
doc = nlp2(test_txt)
displacy.render(doc, style="ent")
nlp3 = spacy.load('de')

doc2 = nlp3(test_txt)

displacy.render(doc2, style="ent")
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
for ent in doc2.ents:

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
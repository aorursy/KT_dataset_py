!python -m spacy download pt_core_news_sm
from nltk import word_tokenize

from pathlib import Path

from spacy import displacy

from spacy.util import compounding, minibatch

from string import punctuation

from tqdm.notebook import tqdm

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pt_core_news_sm

import random

import spacy
auxiliadora = pd.read_json('/kaggle/input/real-estates/auxiliadora.json')

guarida = pd.read_json('/kaggle/input/real-estates/guarida.json')



auxiliadora['real_state'] = 'Auxiliadora Predial'

guarida['real_state'] = 'Guarida'



df = auxiliadora.append(guarida, ignore_index=True, sort=True)



df
by_real_state = df.groupby('real_state')['id'].agg('count')



by_real_state
by_real_state.plot.bar()
df['description'].apply(lambda x: len(x)).mean()
sample_size = len(df)

descriptions = df['description']

stopwords = list(punctuation)

counter = 0



with tqdm(total=sample_size) as pbar:

    for description in descriptions:

        tokens = word_tokenize(description, language='portuguese')

        counter += len([t for t in tokens if t not in stopwords])

        

        pbar.update(1)



counter / sample_size
df['description'] = df['description'].str.replace(r'\s+', ' ').str.strip()
pattern = r'\b(?:pr[oó]xim|frente [aàá]|ao lado d|junto [aàá]|perto d)'

descriptions = df[df['description'].str.contains(pattern)].sample(100, random_state=42)['description']



print('\n\n'.join(descriptions))
pois = [

    'Hospital de Clínicas',

    'Parque da Redenção',

    'Parque Redenção',

    'Zaffari',

    'PUC- RS',

    'Zaffari Higienópolis',

    'shopping lindóia',

    'strip center',

    'bourboun wallig',

    'aeroporto',

    'shopping Lindóia',

    'PUC',

    'colégio Anchieta',

    'Clube União',

    'praça da Encol',

    'Shopping Iguatemi',

    'Igreja Mont\'Serrat',

    'shopping cassol',

    'Puc',

    'Hospital da PUC',

    'Bourbon Ipiranga',

    'Colégio Santo Antônio',

    'Zaffari Ipiranga',

    'Barra Shopping Sul',

    'Big Shop',

    'Hípica',

    'Prado',

    'Museu Iberê Camargo',

    'Píer do Catamarã',

    'Zaffari da Riachuelo',

    'Supermercado Nacional',

    'Barra Shopping',

    'Hipódromo',

    'Carrefour',

    'Iguatemi',

    'Bourbon Country',

    'Wallig',

    'Shopping Iguatemi',

    'Viaduto Otávio Rocha',

    'Praça da Matriz',

    'Teatro São Pedro',

    'Catedral Metropolitana',

    'Palácio Piratini',

    'Bourbon Country',

    'horpital Cristo Redentor',

    'UFRGS',

    'Santa Casa',

    'Parque da Redenção',

    'Sogipa',

    'Hospital Mãe de Deus',

    'Hospital de Clínicas',

    'Hospital Conceição',

    'Iguatemi',

    'Bourbon Wallig',

    'Bourbon Country',

    'Zaffari Higienópolis',

    'faculdade de administração da UFRGS',

    'escola Técnica Parobé',

    'Gasômetro',

    'Centro Administrativo do Estado',

    'Palácio do Governo',

    'Sogipa',

    'UFRGS',

    'Parcão',

    'Mãe de Deus',

    'bourbon Country',

    'Shopping iguatemi',

    'parque germânia',

    'Zaffari',

    'Shopping Lindóia',

    'Puc',

    'Sogipa',

    'Zaffari',

    'Confeitaria Barcelona',

    'Parque da Redenção',

    'Zaffari',

    'Redenção',

    'Parque Germânia',

    'Shopping Iguatemi',

    'Leopoldina Juvenil',

    'G.N. União',

    'Parcão',

    'Bourbon Ipiranga',

    'PUC',

    'CEEE',

    'Renner',

    'Zaffari',

    'Bourbon Shopping Wallig',

    'Parque da Redenção',

    'Praça da Encol',

    'Grêmio Náutio União',

    'Shopping Bourbon Assis Brasil',

    'Parcão',

    'Shopping Moinhos de Vento',

    'Colégio La Salle Nossa Senhora das Dores',

    'Nacional da Lucas de oliveira',

    'Shopping Bourbon Wallig',

    'São Judas Tadeu',

    'Colégio Dom Diogo de Souza',

    'Igreja Cristo Redentor',

    'Hospital Cristo Redentor',

    'Lindóia Shopping',

    'Lindóia Tenis Clube',

    'Hospital Vila Nova',

    'Banrisul',

    'Barra Shopping',

    'PUC',

    'Shopping Bourbon Ipiranga',

    'UFRGS',

    'Zaffari',

    'supermercado Maxxi Atacado',

    'Arena do Grêmio',

    'Strip Center',

    'Cassol',

    'Shopping Lindóia',

    'Bourbon Wallig',

    'Hospital de Clinicas',

    'campus Saúde da UFRGS',

    'Redenção',

    'Parque Moinhos',

    'Iguatemi',

    'ESPM',

]



len(pois)
# All lowercase, no duplicates

pois = list(set([p.lower() for p in pois]))



len(pois)
def get_train_data(descriptions, pois):

    TRAIN_DATA = []

    

    # Biggest strings first

    pois.sort(key=len, reverse=True)



    for description in descriptions:  

        entities = []



        for poi in pois:

            start = description.lower().find(poi)



            if start > -1:

                end = start + len(poi)

                is_overlap = False



                for older_start, older_end, *_ in entities:               

                    # Check if it current POI overlaps another one (eg.: PUCRS and PUC)

                    if start <= older_end and end >= older_start:

                        is_overlap = True



                        break



                if not is_overlap:

                    entities.append((start, end, 'POI'))



        if len(entities) > 0:

            train_instance = (description, {'entities': entities})



            TRAIN_DATA.append(train_instance)

            

    return TRAIN_DATA
TRAIN_DATA = get_train_data(descriptions, pois)



len(TRAIN_DATA), TRAIN_DATA
def train_ner(TRAIN_DATA, n_iter=100):

    nlp = pt_core_news_sm.load()

    

    # create the built-in pipeline components and add them to the pipeline

    # nlp.create_pipe works for built-ins that are registered with spaCy

    if 'ner' not in nlp.pipe_names:

        ner = nlp.create_pipe('ner')

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe('ner')



    # add labels

    for _, annotations in TRAIN_DATA:

        for ent in annotations.get('entities'):

            ner.add_label(ent[2])



    # get names of other pipes to disable them during training

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    with nlp.disable_pipes(*other_pipes):  # only train NER

        with tqdm(total=n_iter) as pbar:

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

                print('Losses', losses)



                pbar.update(1)

        

    return nlp
nlp = train_ner(TRAIN_DATA)
test_descriptions = df['description']

found_pois = set()



with tqdm(total=len(test_descriptions)) as pbar:

    for description in test_descriptions:

        doc = nlp(description)

        description_pois = [e.text.lower() for e in doc.ents if e.label_ == 'POI']



        found_pois.update(description_pois)

        pbar.update(1)
len(found_pois), found_pois
# samples = df[df['description'].str.lower().str.contains('schop')]['description']

samples = df.sample(1000)['description']

examples_count = 10



with tqdm(total=examples_count) as pbar:

    for description in samples:

        doc = nlp(description)

        

        if len([e for e in doc.ents if e.label_ == 'POI']) > 0:

            displacy.render(doc, style='ent')

            pbar.update(1)

            

            examples_count -= 1

            

            if examples_count == 0:

                break
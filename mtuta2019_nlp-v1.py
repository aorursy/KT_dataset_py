import pandas as pd

import spacy
!python -m spacy download es_core_news_md
# Importar vocabulario de Spacy, removiendo del pipeline el NER

nlp = spacy.load('es_core_news_md', disable=['ner'])
def concat_text(pdSeries):

    pdSeries = pdSeries.str.rstrip('.')

    return pdSeries.str.cat(sep='. ')
nombre_archivo = '../input/REPOSITORIO_LECCIONES APRENDIDAS.xlsx'



data = pd.read_excel(nombre_archivo, encoding='latin-1', keep_default_na= False, na_values=[""])

data["CONTEX_LECC"] = [concat_text(i[1]) for i in data[['CONTEXTO', 'LECCIONES APRENDIDAS']].iterrows()]

data.CONTEX_LECC = data.CONTEX_LECC.str.replace('\n', ' ').replace('\s+', ' ')



data.head()
cont_lecc = concat_text(data["CONTEX_LECC"])
word = []

lemma = []

shape = []

pos = []

istop = []

dep = []

head = []

children = []



nlp_text=nlp(cont_lecc)



for token in nlp_text:

    word.append(token.text)

    lemma.append(token.lemma_)

    shape.append(token.shape_)

    pos.append(token.pos_)

    istop.append(token.is_stop)

    dep.append(token.dep_)

    head.append(token.head.text)

    children.append([child for child in token.children])



    

results = pd.DataFrame({'Word':word, 'Lemma':lemma, 'POS':pos, 'DEP':dep, 'head':head,

                             'children':children, 'Shape':shape, 'is_stop':istop})

results.head()
info = []



for possible_subject in nlp_text:

    # si el POS del HEAD de la palabra es VERB y su dependency parsing es nsubj (sujeto nominal)

    if possible_subject.head.pos_ == 'VERB' and possible_subject.dep_ == 'nsubj' :

        children = []

        for child in possible_subject.children:

            # si las ramas son nmod (modificador nominal) y no es espacio

             if child.dep_ in ('nmod') and child.pos_ != 'SPACE': 

                children.append(child)

            

        if children:

            info.append((possible_subject.head.lemma_.lower(),possible_subject.lemma_.lower(),children))

result = pd.DataFrame(info, columns = ['Head' , 'Word', 'Children'])

result.head()
# !!!

info = []

for possible_subject in nlp_text:

    # Si el POS del HEAD de la palabra es VERB y el POS de la palabra es un sustantivo (PROPN y NOUN) y

    # su dependency parsing es sujeto nominal (nsubj)

    if possible_subject.head.pos_ == 'VERB' and possible_subject.pos_ in ('PROPN','NOUN') and possible_subject.dep_=='nsubj':

        info.append((possible_subject.head,possible_subject,possible_subject.lemma_))

        

result_subj = pd.DataFrame(info, columns = ['Head','Word', 'Lemma'])

result_subj.head()
info = []

for possible_subject in nlp_text:

    # Si el POS del HEAD de la palabra es VERB y el POS de la palabra es un sustantivo (PROPN y NOUN) y

    # su dependency parsing es sujeto nominal (nsubj)

    if possible_subject.head.pos_ == 'VERB' and possible_subject.pos_ in ('PROPN','NOUN') and possible_subject.dep_=='nsubj':

        children = []

        for child in possible_subject.children:

            # Solo agregar si no es identificado como espacio

            if child.pos_ != 'SPACE':

                children.append(child)

            

        if children:

            info.append((possible_subject.head.lemma_,possible_subject,possible_subject.lemma_.lower(),children))

            

result_subj1 = pd.DataFrame(info, columns = ['Head' , 'Word', 'Lemma', 'Children'])
info = []

for possible_subject in nlp_text:

    if possible_subject.pos_ == 'VERB' and possible_subject.dep_=='nsubj':

        children = []

        for child in possible_subject.children:

            # Solo agregar si no es identificado como espacio

            if child.pos_ != 'SPACE':

                children.append(child)

            

        if children:

            info.append((possible_subject.head.lemma_,possible_subject,possible_subject.lemma_.lower(),children))

            

result_subj2 = pd.DataFrame(info, columns = ['Head' , 'Word', 'Lemma', 'Children'])

result_subj2.head()
print(result_subj2.Lemma.value_counts().nlargest(10, keep='all'))
from __future__ import unicode_literals

import textacy

from collections import defaultdict



###

# Patrón para extraer información de un texto basado en reglas(con expresiones regulares basados en token).

###



patron = r'<PROPN>+ (<PUNCT|CCONJ> <PUNCT|CCONJ>? <PROPN>+)*'

param = []

i = 0

while i < len(data["CONTEX_LECC"]):

    lists_ = []

    sent = nlp(data["CONTEX_LECC"].iloc[i])

    doc = textacy.make_spacy_doc(sent, lang='es_core_news_md')

    lists_ = textacy.extract.pos_regex_matches(doc, patron)

    for item in lists_:

        if len(item) != 0:

            param.append(item.text.lower())

    i +=1



j=0

aux = defaultdict(list)

for index, item in enumerate(param):

    aux[item].append(index)



result = {item: len(indexs) for item, indexs in aux.items() if len(indexs) >= 1}

key = list(result.keys())

key.sort()
#Normalizar los datos del diccionario de entidades

key = list(result.keys())

print(len(key))

for item in key:

    i = 0

    key_new = []

    key_new = key

    key_new.remove(item)

    while i < len(key) - 1:        

#         print(len(key_new))

        i += 1
info = []



for possible_subject in nlp_text:

    # si el POS del HEAD de la palabra es VERB y su dependency parsing es nsubj (sujeto nominal)

    if possible_subject.head.pos_ == 'VERB' and possible_subject.dep_ == 'nsubj' :

        children = []

        for child in possible_subject.children:

            # si las ramas son nmod (modificador nominal) y no es espacio

            if str(child).lower() in key:

                 if child.dep_ in ('nmod') and child.pos_ != 'SPACE':

                        info.append((possible_subject.text.lower(),possible_subject.head.lemma_.lower(),child.text.lower()))



result_ent = pd.DataFrame(info, columns = ['Word', 'Head', 'Children'])

result_ent
result_ent[result_ent.Children == 'asesoftware']
info = []

for possible_subject in nlp_text:

    if possible_subject.head.pos_ == 'VERB' and possible_subject.pos_ in ('PROPN','NOUN'):

        children = []

        for child in possible_subject.children:

            # Solo agregar si no es identificado como espacio

            if str(child).lower() in key:

                if child.pos_ != 'SPACE':

                    info.append((possible_subject.text.lower(),possible_subject.head.lemma_.lower(),child.text.lower()))

                                    

result_ent1 = pd.DataFrame(info, columns = ['Word', 'Head', 'Children'])

result_ent1
result_ent1[result_ent1.Children == 'proyecto']
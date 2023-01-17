!pip install neuralcoref
!python -m spacy download en
import spacy
nlp = spacy.load('en')

import neuralcoref
neuralcoref.add_to_pipe(nlp)

doc = nlp(u'My sister has a dog. She loves him.')

doc._.has_coref
doc._.coref_clusters
doc = nlp(u'My sister has a dog. She loves him but he is very silly. Once a cat crossed his way and he started fleeing in fear.')

doc._.has_coref
doc._.coref_clusters
doc = nlp(u'My sister has a dog. She loves him but he is very silly. \
Once a cat crossed his way and he started fleeing in fear. At the end, the poor kitty was just trying to find some food nearby.')

doc._.has_coref
doc._.coref_clusters
doc = nlp(u'Mark Johnson has recently announced he will be attending to patients at St. Paul hospital from May 3rd. \
The hospital team stated they are very honnored on having him -- a renowned cardiologist -- as part of their staff.')

doc._.has_coref
doc._.coref_clusters

for ent in doc.ents:
    print(ent.text, ent.label_)
    print(ent._.coref_cluster)
import nltk

nltk.download('reuters')
from nltk.corpus import reuters
#print(reuters.raw('test/14826'))
text_reuters = reuters.raw('test/14826')
sent_text = nltk.sent_tokenize(text_reuters)
sent_text[2]
#doc = nlp(sent_text[2])
doc = nlp(text_reuters)

doc._.has_coref
doc._.coref_clusters
#Organização ('ORG'), Pessoa ('PERSON') e Localidade ('LOC' ou 'GPE'),
def unique_elements(trends):
    output = []
    for x in trends:
        if x not in output:
            output.append(x)
    return output

org = []
per = []
loc = []

for ent in doc.ents:
    entity_name = ent.text
    entity_label = ent.label_
    #print(ent.text, ent.label_)
    #print(ent._.coref_cluster)
    #print('======================')
    if entity_label == 'ORG':
        org.append(entity_name.replace('\n', ''))
    if entity_label == 'PERSON':
        per.append(entity_name.replace('\n', ''))
    if entity_label == 'LOC' or entity_label == 'GPE':
        loc.append(entity_name.replace('\n', ''))
print('Lista de Organizações: ')
print(unique_elements(org))

print('Lista de Pessoas: ')
print(unique_elements(per))
print('Lista de Locais: ')
print(unique_elements(loc))

'''

This is the sixth homework of NLP course.



The homework is depends on the student's request. I wanted to try build a simple chatbot. 

'''



'''

Hocam şuan ki bitirme tezim bu kodda kullanacağım atis database i kapsadığından, 

sınıflandırıcıyı tamamladığım zaman chatbot içinde gerekli dialogue manager ve natural

language generator ı sağlayıp bu projeyi de bitirmeyi düşünüyorum. Eğer yapabilirsem

son halini yine sizinle paylaşacağım. Gerekli yönlendirmeleri verdiğiniz için teşekkür ederim.

'''

import spacy                # To build a nlu, I prefer spacy to extract required information

from spacy import displacy

import pandas as pd

from spacy.tokenizer import Tokenizer

from spacy.lang.en import English
nlp = spacy.load("en")

tokenizer = Tokenizer(nlp.vocab)
sentence = 'I would like to go to from Atlanta to New York tomorrow with plane.'



doc = nlp(sentence)

for token in doc:

    print(token.text, "-->\n    Lemma:", token.lemma_, " \n Pos:", token.pos_," \n Tag:" , token.tag_)

    print("________________________________________________________")
print("children of ", doc[7])

print([token.text for token in doc[7].children])



print("\nlefts of ", doc[7])

print([token.text for token in doc[7].lefts])



print("\nrights of ", doc[7])

print([token.text for token in doc[7].rights])



print("\nsubtrees of ", doc[7])

print (list(doc[7].subtree))
print("Nouns:")



for chunk in doc.noun_chunks:

     print (chunk)

        

# verb için pip install textacy
for ent in doc.ents:

     print(ent.text, ent.start_char, ent.end_char,

           ent.label_, spacy.explain(ent.label_))
for entity in doc.ents:

    print(f"{entity.text} ({entity.label_})") # Named entities
# displacy.serve(doc, style="ent")   # Delete the "#" to visualize the NER
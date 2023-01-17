import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp(u"John Adam is one the researcher who invent the direction of way towards success ")

for token in doc:
    print(token.text , '\t' , token.pos_ , '\t' , token.lemma , '\t' , token.lemma_)
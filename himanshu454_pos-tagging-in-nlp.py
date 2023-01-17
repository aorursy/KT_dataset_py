import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp("Apple is looking at buying U.K. startup for $1 Billion")
print("Text \t\t Lemma \t\t POS \t\t is_Stop_word")
for token in doc:
    print(token.text ,"\t\t",token.lemma_,"\t\t",token.pos_,"\t\t",token.is_stop,"\n")
from spacy import displacy
doc = nlp("The quick brown fox jumped over the lazy dog's Back")
displacy.render(doc , style = "dep" , jupyter = True , options = {'distance' : 100})
import re


#split by WhiteSpace
text = "I'm with you for the entire life in U.K..!"
words = re.split(r'\W+' , text)
print(words)
#using Spacy

import spacy
nlp = spacy.load('en_core_web_sm')
string = "I'm with you for the entire life in U.K..!"
string
doc = nlp(string)
for i in doc:
    print(i.text , end = " | ")
doc3 = nlp(u"A 5km NYC cab ride costs $10.20")
for t in doc3:
    print(t)
len(doc3.vocab)
from spacy import displacy
doc = nlp(u"Apple is going to built a U.K. Factory for $3")
displacy.render(doc , style = 'dep' , jupyter = True , options = {"distance" : 80})
#importing lib

import spacy
import en_core_web_sm
from spacy import displacy

text='Trump claims that he had no choice but to risk his own health. Americans disagree. and Infectious Trump briefly leaves hospital to greet supporters. also  Positive tests for senators raise doubts about fast-track confirmation of Trump’s Supreme Court choice'

#loading text 
nlp = en_core_web_sm.load()

doc = nlp(text)

#making sentances tokenzation ...coverting text to seperate sentances 
for sentances in doc.sents:
    print(sentances)
  
#making tokenization for words .. converting text to words     
tok_words=[]
for word in doc:
    tok_words.append(word.text)
    
print(tok_words)
len(tok_words)

#check stop words 
StopWords=spacy.lang.en.stop_words.STOP_WORDS
print(StopWords)
len(StopWords)

#removing stop words as it is not important 
filtered_words=[]

for w in tok_words:
    if w not in StopWords:
        filtered_words.append(w)
        
print(filtered_words)
len(filtered_words)

#checking tag and pos

for token in doc:
    print((token.text,token.pos_,token.tag_))
    

#detecting Nouns

for Noun in doc.noun_chunks:
    print(Noun)
#NER

for ent in doc.ents:
    print(ent.text,ent.label_)
for i in doc.ents:
    (i,i.label_,i.label)
displacy.render(doc,style='ent',jupyter=True)
nlp.pipe_names

for token in doc:
    print(token.text, "-->", token.dep_)
spacy.explain("nsubj"), spacy.explain("ROOT"), spacy.explain("aux"), spacy.explain("advcl"), spacy.explain("dobj")
import spacy
nlp = spacy.load('en_core_web_sm')

# Import spaCy Matcher
from spacy.matcher import Matcher

# Initialize the matcher with the spaCy vocabulary
matcher = Matcher(nlp.vocab)

doc = nlp("Some people start their day with lemon water")

# Define rule
pattern = [{'TEXT': 'lemon'}, {'TEXT': 'water'}]

# Add rule
matcher.add('rule_1', None, pattern)
matches = matcher(doc)
matches
# Extract matched text
for match_id, start, end in matches:
    # Get the matched span
    matched_span = doc[start:end]
    print(matched_span.text)

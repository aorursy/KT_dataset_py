import spacy

nlp = spacy.load('en_core_web_sm')
doc = nlp("Tea is healthy, calming, and delicious, don't you think?")
for token in doc:

    print(token)
print("{:<15}{:<15}{}".format('Token', 'Lemma', 'Stopword'))

print("-"*40)

for token in doc:

    print("{:<15}{:<15}{}".format(str(token), token.lemma_, token.is_stop))
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']

patterns = [nlp(text) for text in terms]

matcher.add("TerminologyList", None, *patterns)
# Borrowed from https://daringfireball.net/linked/2019/09/21/patel-11-pro

text_doc = nlp("Glowing review overall, and some really interesting side-by-side photography "

               "tests pitting the iPhone 11 Pro against the Galaxy Note 10 Plus and last year’s " 

               "iPhone XS and Google Pixel 3.") 

matches = matcher(text_doc)

print(matches)
match_id, start, end = matches[0]

print(nlp.vocab.strings[match_id], text_doc[start:end])
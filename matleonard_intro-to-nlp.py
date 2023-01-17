import spacy

nlp = spacy.load('en')
doc = nlp("Tea is healthy and calming, don't you think?")
for token in doc:

    print(token)
print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))

print("-"*40)

for token in doc:

    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']

patterns = [nlp(text) for text in terms]

matcher.add("TerminologyList", patterns)
# Borrowed from https://daringfireball.net/linked/2019/09/21/patel-11-pro

text_doc = nlp("Glowing review overall, and some really interesting side-by-side "

               "photography tests pitting the iPhone 11 Pro against the "

               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 

matches = matcher(text_doc)

print(matches)
match_id, start, end = matches[0]

print(nlp.vocab.strings[match_id], text_doc[start:end])
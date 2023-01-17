import spacy

nlp = spacy.load('en')
doc = nlp("Tea is healthy and calming, don't you think?")
for token in doc:

    print(token)
from spacy.matcher import PhraseMatcher

matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']

patterns = [nlp(text) for text in terms]

matcher.add("TerminologyList", None, *patterns)
# Borrowed from https://daringfireball.net/linked/2019/09/21/patel-11-pro

text_doc = nlp("Glowing review overall, and some really interesting side-by-side "

               "photography tests pitting the iPhone 11 Pro against the "

               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.") 

matches = matcher(text_doc)

print(matches)
match_id, start, end = matches[0]

print(nlp.vocab.strings[match_id], text_doc[start:end])
import pandas as pd

# Loading the spam data

spam = pd.read_csv('../input/nlp-course/spam.csv')

# Read data

spam.head(10)
import spacy



# Create an empty model

nlp = spacy.blank('en')



# Create the TextCategorizer with exclusive classes and "bow" architecture

textcat = nlp.create_pipe("textcat", config={"exclusive_classes": True, "architecture":"bow"})



# Add the TextCategorizer to the empty model

nlp.add_pipe(textcat)
# Add labels to text classifier

textcat.add_label("ham")

textcat.add_label("spam")
train_texts = spam['text'].values

train_labels = [{'cats':{'ham':label == 'ham', 'spam':label == 'spam'}}

               for label in spam['label']]
train_data = list(zip(train_texts, train_labels))

train_data[:3]
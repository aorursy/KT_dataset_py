import pandas as pd
import nltk

# tagged corpus in portuguese
from nltk.corpus import mac_morpho, floresta, machado
# 1. Train a POS Tagger for Portuguese
training_size = int(len(mac_morpho.tagged_sents()) * 0.9)
training_sentences = mac_morpho.tagged_sents()[:training_size]

tagger0 = nltk.tag.UnigramTagger(training_sentences)
tagger1 = nltk.tag.BigramTagger(training_sentences, backoff=tagger0)
tagger = nltk.tag.TrigramTagger(training_sentences, backoff=tagger1)
# 2. Apply to some new text
comments = pd.read_csv('../input/comments.csv')
text = comments.loc[0].comentario
tokens = nltk.word_tokenize(text.lower(), language='portuguese')
print(tagger.tag(tokens))
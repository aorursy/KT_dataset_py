import nltk
from nltk.corpus import gutenberg

gutenberg.fileids()
emma = gutenberg.words('austen-emma.txt')
print(len(emma))
print(emma)
emma_text = nltk.Text(gutenberg.words('austen-emma.txt'))
print(emma_text,len(emma_text))
for fileid in gutenberg.fileids():
    num_chars = len(gutenberg.raw(fileid))
    num_words = len(gutenberg.words(fileid))
    num_sents = len(gutenberg.sents(fileid))
    num_vocab = len(set([w.lower() for w in gutenberg.words(fileid)]))
    print(fileid,num_chars/num_words,num_chars/num_sents,num_words/num_vocab)
    
macbeth_sentences = gutenberg.sents('shakespeare-macbeth.txt')
print(macbeth_sentences)
print(len(macbeth_sentences))
macbeth_sentences[1037]
longest_len = max([len(s) for s in macbeth_sentences])
print(longest_len)
longest_sentences = [s for s in macbeth_sentences if(len(s)==longest_len)]
print(len(longest_sentences))
print(longest_sentences)

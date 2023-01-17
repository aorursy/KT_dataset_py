# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output





import nltk

from nltk.corpus import brown

from nltk import word_tokenize, pos_tag

text = 'I do not like green eggs and ham, I do not like them Sam I am!'

tokens = word_tokenize(text)

print("My text : ", text)

print("My tokens : ", tokens)
brown_tagged_sents = brown.tagged_sents(categories='news')

brown_sents = brown.sents(categories='news')
"""

Search the max tag in Brown Corpus

"""

tags = [tag for (word, tag) in brown.tagged_words(categories='news')]

print("Most common tag is : ", nltk.FreqDist(tags).max())



"""

Now we can create a tagger that tags everything as NN

"""

# Default Tagging

default_tagger = nltk.DefaultTagger('NN')

print("\nCheck results : ", default_tagger.tag(tokens))



# Performances : 

print("\nPerformance with default tagger : ", default_tagger.evaluate(brown_tagged_sents))
from nltk import word_tokenize, pos_tag



# Pos-Tagging

pos_tagger = nltk.pos_tag(tokens)

print("With POS_TAG : ", pos_tagger)
text = 'all your base are belong to us all of your base base base'

type(text)
"""

Define your pattern

"""

patterns = [

    (r'.*ing$', 'VBG'),               # gerunds

    (r'.*ed$', 'VBD'),                # simple past

    (r'.*es$', 'VBZ'),                # 3rd singular present

    (r'.*ould$', 'MD'),               # modals

    (r'.*\'s$', 'NN$'),               # possessive nouns

    (r'.*s$', 'NNS'),                 # plural nouns

    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers

    (r'(The|the|A|a|An|an)$', 'AT'),   # articles 

    (r'.*able$', 'JJ'),                # adjectives 

    (r'.*ness$', 'NN'),                # nouns formed from adjectives

    (r'.*ly$', 'RB'),                  # adverbs

    (r'(He|he|She|she|It|it|I|me|Me|You|you)$', 'PRP'), # pronouns

    (r'(His|his|Her|her|Its|its)$', 'PRP$'),    # possesive

    (r'(my|Your|your|Yours|yours)$', 'PRP$'),   # possesive

    (r'(on|On|in|In|at|At|since|Since)$', 'IN'),# time prepopsitions

    (r'(for|For|ago|Ago|before|Before)$', 'IN'),# time prepopsitions

    (r'(till|Till|until|Until)$', 'IN'),        # time prepopsitions

    (r'(by|By|beside|Beside)$', 'IN'),          # space prepopsitions

    (r'(under|Under|below|Below)$', 'IN'),      # space prepopsitions

    (r'(over|Over|above|Above)$', 'IN'),        # space prepopsitions

    (r'(across|Across|through|Through)$', 'IN'),# space prepopsitions

    (r'(into|Into|towards|Towards)$', 'IN'),    # space prepopsitions

    (r'(onto|Onto|from|From)$', 'IN'),          # space prepopsitions    

    (r'\.$','.'), (r'\,$',','), (r'\?$','?'),    # fullstop, comma, Qmark

    (r'\($','('), (r'\)$',')'),             # round brackets

    (r'\[$','['), (r'\]$',']'),             # square brackets

    (r'(Sam)$', 'NAM'),

    # WARNING : Put the default value in the end

    (r'.*', 'NN')                      # nouns (default)

    ]



"""

Construct tager

"""

regexp_tagger = nltk.RegexpTagger(patterns)



# We use the sentence : brown_sents[3]

print(regexp_tagger.tag(brown_sents[3]))

print(regexp_tagger.evaluate(brown_tagged_sents))



# We use our sentence :

print(regexp_tagger.tag(tokens))

print(regexp_tagger.evaluate(brown_tagged_sents))
"""

UniGram-Tagging

"""

from nltk.corpus import brown



# Training

unigram_tagger = nltk.UnigramTagger(brown_tagged_sents)



# Tag our text

unigram_tagger.tag(tokens)



# Evaluate 

unigram_tagger.evaluate(brown_tagged_sents)
"""

Train your own Unigram 

"""

# Create a train and test set

size = int(len(brown_tagged_sents) * 0.9)

train_sents = brown_tagged_sents[:size]

test_sents = brown_tagged_sents[size:]



# Training : 

unigram_tagger = nltk.UnigramTagger(train_sents)



# Evaluate

print ("Evaluation 1gram on train set ", unigram_tagger.evaluate(train_sents))

print ("Evaluation 1gram on test set ", unigram_tagger.evaluate(test_sents))
"""

BiGram-Tagging

"""

# Training the bigram tagger on a train set

bigram_tagger = nltk.BigramTagger(brown_tagged_sents)



# Tag our text

bigram_tagger.tag(tokens)



# Evaluate 

bigram_tagger.evaluate(brown_tagged_sents)
"""

Train your own Bigram 

"""

# Create a train and test set

size = int(len(brown_tagged_sents) * 0.9)

train_sents = brown_tagged_sents[:size]

test_sents = brown_tagged_sents[size:]



# Training the bigram tagger on a train set

bigram_tagger = nltk.BigramTagger(train_sents)



# Evaluate

print ("Evaluation 2gram on train set ", bigram_tagger.evaluate(train_sents))

print ("Evaluation 2gram on test set ", bigram_tagger.evaluate(test_sents))
"""

TriGram-Tagging

"""

# Training the bigram tagger on a train set

Trigram_tagger = nltk.TrigramTagger(brown_tagged_sents)



# Tag our text

Trigram_tagger.tag(tokens)



# Evaluate 

Trigram_tagger.evaluate(brown_tagged_sents)
"""

Train your own Trigram 

"""

# Create a train and test set

size = int(len(brown_tagged_sents) * 0.9)

train_sents = brown_tagged_sents[:size]

test_sents = brown_tagged_sents[size:]



# Training the bigram tagger on a train set

Trigram_tagger = nltk.TrigramTagger(train_sents)



# Evaluate

print ("Evaluation 3gram on train set ", Trigram_tagger.evaluate(train_sents))

print ("Evaluation 3gram on test set ", Trigram_tagger.evaluate(test_sents))
"""

Mix Default, Unigram and Bigram

"""

t0 = nltk.DefaultTagger('NN')

t1 = nltk.UnigramTagger(train_sents, backoff=t0)

t2 = nltk.BigramTagger(train_sents, backoff=t1)



print ("Evaluation mix default/1G/2G on train set ", t2.evaluate(train_sents))

print ("Evaluation mix default/1G/2G on test set ", t2.evaluate(test_sents))



"""

Combine Default, Unigram and Bigram

"""

t0 = nltk.DefaultTagger('NN')

t1 = nltk.UnigramTagger(train_sents, backoff=t0)

t2 = nltk.BigramTagger(train_sents, backoff=t1)

t3 = nltk.TrigramTagger(train_sents, backoff=t2)

print ("\nEvaluation mix default/1G/2G/3G on train set ", t3.evaluate(train_sents))

print ("Evaluation mix default/1G/2G/3G on test set ", t3.evaluate(test_sents))

# Create a train and test set

size = int(len(brown_tagged_sents) * 0.9)

train_sents = brown_tagged_sents[:size]

test_sents = brown_tagged_sents[size:]



# Train the model 

from nltk.tag.perceptron import PerceptronTagger

pct_tag = PerceptronTagger(load=False)

pct_tag.train(train_sents)



# Check the performance 

print ("Evaluation Own PerceptronTagger on train set ", pct_tag.evaluate(train_sents))

print ("Evaluation Own PerceptronTagger on test set ", pct_tag.evaluate(test_sents))
""" 

which of these tags are the most common in the news category of the Brown corpus ? 

"""

from nltk.corpus import brown

brown_news_tagged = brown.tagged_words(categories='news', tagset='universal')

tag_fd = nltk.FreqDist(tag for (word, tag) in brown_news_tagged)

print("List of most common tags in Brown corpus : \n", tag_fd.most_common())

tag_fd.plot(cumulative=True)



"""

Rechercher des tags specifiques """

def find_tags(tag_prefix, tokens):

    return [tokens for tokens, pos in pos_tag(tokens) if pos == tag_prefix]

mytag = find_tags("NNP", tokens)

print("Les tags sont : ", mytag)
print("Modern Chinese ", nltk.corpus.sinica_treebank.tagged_words())

print("Indian : " , nltk.corpus.indian.tagged_words())

print("Portuguese : ", nltk.corpus.mac_morpho.tagged_words())

print("Brasil : ", nltk.corpus.conll2002.tagged_words())

print("Catalan : ", nltk.corpus.cess_cat.tagged_words())
""" STORE TAGGERS """

# save our tagger t2 

from cPickle import dump

output = open('t2.pkl', 'wb')

dump(t2, output, -1)

output.close()



# we can load our saved tagger

from cPickle import load

input = open('t2.pkl', 'rb')

tagger = load(input)

input.close()



# Check 

text = """The board's action shows what free enterprise is up against in our complex maze of regulatory laws ."""

tokens = text.split()

tagger.tag(tokens)
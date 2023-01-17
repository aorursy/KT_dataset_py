import re



from nltk.corpus import (LazyCorpusLoader, CategorizedPlaintextCorpusReader,

                         CategorizedTaggedCorpusReader, BracketParseCorpusReader,

                         CategorizedBracketParseCorpusReader, PropbankCorpusReader,

                        NombankCorpusReader, SwadeshCorpusReader,

                        TimitCorpusReader, TimitTaggedCorpusReader,

                        SemcorCorpusReader, KNBCorpusReader)

import nltk

# Removing the original path

if '/usr/share/nltk_data' in nltk.data.path:

    nltk.data.path.remove('/usr/share/nltk_data')

nltk.data.path.append('../input/')

nltk.data.path
movie_reviews = LazyCorpusLoader('movie_reviews', CategorizedPlaintextCorpusReader,

                                 r'(?!\.).*\.txt', cat_pattern=r'(neg|pos)/.*',

                                 encoding='ascii', 

                                 nltk_data_subdir='movie-review/movie_reviews/')



documents = [([w for w in movie_reviews.words(i)],

              i.split('/')[0]) for i in movie_reviews.fileids()]



documents[0][1], ' '.join(documents[0][0][:10])
brown = LazyCorpusLoader('brown', CategorizedTaggedCorpusReader, r'c[a-z]\d\d',

                         cat_file='cats.txt', tagset='brown', encoding="ascii",

                        nltk_data_subdir='brown-corpus/brown')



brown.tagged_sents()
cess_cat = LazyCorpusLoader('cess_cat', BracketParseCorpusReader, r'(?!\.).*\.tbf',

                            tagset='unknown', encoding='ISO-8859-15',

                            nltk_data_subdir='cess-treebanks/cess_cat')



cess_esp = LazyCorpusLoader('cess_esp', BracketParseCorpusReader, r'(?!\.).*\.tbf',

                            tagset='unknown', encoding='ISO-8859-15',

                            nltk_data_subdir='cess-treebanks/cess_esp')



cess_cat.tagged_sents()

cess_esp.tagged_sents()
treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg',

                            tagset='wsj', encoding='ascii', 

                            nltk_data_subdir='penn-tree-bank/treebank')



treebank.tagged_sents()
# Must be defined *after* treebank corpus.

propbank = LazyCorpusLoader('propbank', PropbankCorpusReader,

                            'prop.txt', 'frames/.*\.xml', 'verbs.txt',

                            lambda filename: re.sub(r'^wsj/\d\d/', '', filename), treebank,

                            nltk_data_subdir='propbank/propbank') 

propbank.instances()
# Must be defined *after* treebank corpus.

nombank = LazyCorpusLoader('nombank.1.0', NombankCorpusReader, 

                           'nombank.1.0', 'frames/.*\.xml', 'nombank.1.0.words',

                           lambda filename: re.sub(r'^wsj/\d\d/', '', filename), treebank,

                           nltk_data_subdir='nombank/nombank') # Must be defined *after* treebank corpus.





#nombank.instances()
swadesh110 = LazyCorpusLoader('panlex_swadesh', SwadeshCorpusReader, 

                              r'swadesh110/.*\.txt', encoding='utf8',

                             nltk_data_subdir='panlex-swadesh/panlex_swadesh')

swadesh110.words()
timit = LazyCorpusLoader('timit', TimitCorpusReader, 

                        nltk_data_subdir='timitcorpus/timit')



timit_tagged = LazyCorpusLoader('timit', TimitTaggedCorpusReader, '.+\.tags',

                                tagset='wsj', encoding='ascii',

                               nltk_data_subdir='timitcorpus/timit')



timit.sents()

timit_tagged.tagged_sents()
reuters = LazyCorpusLoader('reuters', CategorizedPlaintextCorpusReader, 

                           '(training|test).*', cat_file='cats.txt', encoding='ISO-8859-2',

                          nltk_data_subdir='reuters/reuters/')

reuters.words()
knbc = LazyCorpusLoader('knbc/corpus1', KNBCorpusReader, r'.*/KN.*', encoding='euc-jp',

                       nltk_data_subdir='knb-corpus/knbc/')

knbc.words()
# semcor needs the wordnet data.

# Add the path back to load wordnet 

# Removing the original path

nltk.data.path.append('/usr/share/nltk_data')

from nltk.corpus import wordnet



# Lets remove it again...

if '/usr/share/nltk_data' in nltk.data.path:

    nltk.data.path.remove('/usr/share/nltk_data')

    

semcor = LazyCorpusLoader('semcor', SemcorCorpusReader, r'brown./tagfiles/br-.*\.xml',

                          wordnet,  nltk_data_subdir='semcor-corpus/semcor/') # Must be defined *after* wordnet corpus.



semcor.tagged_chunks()
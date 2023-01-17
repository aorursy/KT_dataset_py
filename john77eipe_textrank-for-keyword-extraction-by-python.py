# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from collections import OrderedDict

import spacy

from spacy import displacy

from spacy.lang.en.stop_words import STOP_WORDS

from spacy.lang.en import English

from spacy.matcher import Matcher



nlp = spacy.load('en_core_web_sm')



def custom_retokenizer(doc):

    matcher = Matcher(nlp.vocab)

    patterns = [[{"LOWER": "el"}, {"LOWER": "paso"}]]

    matcher.add("TO_MERGE", None, *patterns)

    matches = matcher(doc)

    with doc.retokenize() as retokenizer:

        for match_id, start, end in matches:

            span = doc[start:end]

            retokenizer.merge(span)

    return doc
nlp.add_pipe(custom_retokenizer, before="tagger")
class TextRankForKeyword():

    """Extract keywords from text"""

    

    def __init__(self):

        self.d = 0.85 # damping coefficient, usually is .85

        self.min_diff = 1e-5 # convergence threshold

        self.steps = 10 # iteration steps

        self.node_weight = None # save keywords and its weight



    

    def set_stopwords(self, stopwords):  

        """Set stop words"""

        for word in STOP_WORDS.union(set(stopwords)):

            lexeme = nlp.vocab[word]

            lexeme.is_stop = True

    

    def sentence_segment(self, doc, candidate_pos, lower):

        """Store those words only in cadidate_pos"""

        sentences = []

        for sent in doc.sents:

            selected_words = []

            for token in sent:

                # Store words only with cadidate POS tag

                if token.pos_ in candidate_pos and token.is_stop is False:

                    if lower is True:

                        selected_words.append(token.text.lower())

                    else:

                        selected_words.append(token.text)

            sentences.append(selected_words)

        return sentences

        

    def get_vocab(self, sentences):

        """Get all tokens"""

        vocab = OrderedDict()

        i = 0

        for sentence in sentences:

            for word in sentence:

                if word not in vocab:

                    vocab[word] = i

                    i += 1

        return vocab

    

    def get_token_pairs(self, window_size, sentences):

        """Build token_pairs from windows in sentences"""

        token_pairs = list()

        for sentence in sentences:

            for i, word in enumerate(sentence):

                for j in range(i+1, i+window_size):

                    if j >= len(sentence):

                        break

                    pair = (word, sentence[j])

                    if pair not in token_pairs:

                        token_pairs.append(pair)

        return token_pairs

        

    def symmetrize(self, a):

        return a + a.T - np.diag(a.diagonal())

    

    def get_matrix(self, vocab, token_pairs):

        """Get normalized matrix"""

        # Build matrix

        vocab_size = len(vocab)

        g = np.zeros((vocab_size, vocab_size), dtype='float')

        for word1, word2 in token_pairs:

            i, j = vocab[word1], vocab[word2]

            g[i][j] = 1

            

        # Get Symmeric matrix

        g = self.symmetrize(g)

        

        # Normalize matrix by column

        norm = np.sum(g, axis=0)

        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm

        

        return g_norm



    

    def get_keywords(self, number=10):

        """Print top number keywords"""

        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))

        for i, (key, value) in enumerate(node_weight.items()):

            print(key + ' - ' + str(value))

            if i > number:

                break

        

        

    def analyze(self, text, 

                candidate_pos=['NOUN', 'PROPN'], 

                window_size=4, lower=False, stopwords=list()):

        """Main function to analyze text"""

        

        # Set stop words

        self.set_stopwords(stopwords)

        

        # Pare text by spaCy

        doc = nlp(text)

        

        for sent in doc.sents:

            for token in sent:

                print(token.text)

        for sent in doc.sents:

            displacy.render(sent, style="dep")    

        for sent in doc.sents:

            displacy.render(sent, style="ent")

        

        # Filter sentences

        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words

        

        # Build vocabulary

        vocab = self.get_vocab(sentences)

        

        # Get token_pairs from windows

        token_pairs = self.get_token_pairs(window_size, sentences)

        print(token_pairs)

        # Get normalized matrix

        g = self.get_matrix(vocab, token_pairs)

        

        # Initialization for weight(pagerank value)

        pr = np.array([1] * len(vocab))

        

        # Iteration

        previous_pr = 0

        for epoch in range(self.steps):

            pr = (1-self.d) + self.d * np.dot(g, pr)

            if abs(previous_pr - sum(pr))  < self.min_diff:

                break

            else:

                previous_pr = sum(pr)



        # Get weight for each node

        node_weight = dict()

        for word, index in vocab.items():

            node_weight[word] = pr[index]

        

        self.node_weight = node_weight
import warnings

warnings.filterwarnings('ignore')
text = '''

A woman in a wedding dress, the Bride, lies wounded in a chapel in El Paso, Texas, having been attacked by the Deadly Viper Assassination Squad. She tells their leader, Bill, that she is pregnant with his baby. He shoots her in the head.

Four years later, having survived the attack, the Bride goes to the home of Vernita Green, planning to kill her.

'''



tr4w = TextRankForKeyword()

tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)

tr4w.get_keywords(10)
"""Example of training spaCy's named entity recognizer, starting off with an

existing model or a blank model.



For more details, see the documentation:

* Training: https://spacy.io/usage/training

* NER: https://spacy.io/usage/linguistic-features#named-entities



Compatible with: spaCy v2.0.0+

Last tested with: v2.1.0

"""

from __future__ import unicode_literals, print_function



import plac

import random

from pathlib import Path

from spacy.util import minibatch, compounding





# training data

TRAIN_DATA = [

    ("Where is El Paso?", {"entities": [(10, 17, "GPE")]}),

    ("I like London and El Paso.", {"entities": [(7, 13, "GPE"), (18, 25, "GPE")]}),

    ("I like to go to the place called El Paso.", {"entities": [(33, 40, "GPE")]}),

    ("El Paso is in Texas.", {"entities": [(0, 7, "GPE"), (14, 29, "GPE")]}),

    ("Krid was born in El Paso, Texas.", {"entities": [(0, 4, "PERSON"), (17, 24, "GPE"), (26, 31, "GPE")]}),

    ("In 1680, the small village of El Paso became the temporary base for Spanish governance", {"entities": [(3, 7, "DATE"), (30, 37, "GPE"), (68, 75, "NORP")]})

]
@plac.annotations(

    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),

    output_dir=("Optional output directory", "option", "o", Path),

    n_iter=("Number of training iterations", "option", "n", int),

)

def retrain(model="en", output_dir=None, n_iter=100):

    """Load the model, set up the pipeline and train the entity recognizer."""

    if model is not None:

        nlp = spacy.load(model)  # load existing spaCy model

        print("Loaded model '%s'" % model)

    else:

        nlp = spacy.blank("en")  # create blank Language class

        print("Created blank 'en' model")



    # create the built-in pipeline components and add them to the pipeline

    # nlp.create_pipe works for built-ins that are registered with spaCy

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner, last=True)

    # otherwise, get it so we can add labels

    else:

        ner = nlp.get_pipe("ner")



    # add labels

    for _, annotations in TRAIN_DATA:

        for ent in annotations.get("entities"):

            ner.add_label(ent[2])



    # get names of other pipes to disable them during training

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    print("Other pipes present: ", other_pipes)

    

    with nlp.disable_pipes(*other_pipes):  # only train NER

        # reset and initialize the weights randomly – but only if we're

        # training a new model

        if model is None:

            nlp.begin_training()

        for itn in range(n_iter):

            random.shuffle(TRAIN_DATA)

            losses = {}

            # batch up the examples using spaCy's minibatch

            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(

                    texts,  # batch of texts

                    annotations,  # batch of annotations

                    drop=0.5,  # dropout - make it harder to memorise data

                    losses=losses,

                )

            print("Losses", losses)



    # test the trained model

    for text, _ in TRAIN_DATA:

        doc = nlp(text)

        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])



    # save model to output directory

    if output_dir is not None:

        output_dir = Path(output_dir)

        if not output_dir.exists():

            output_dir.mkdir()

        nlp.to_disk(output_dir)

        print("Saved model to", output_dir)



        # test the saved model

        print("Loading from", output_dir)

        nlp2 = spacy.load(output_dir)

        for text, _ in TRAIN_DATA:

            doc = nlp2(text)

            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

# retrain(model="en_core_web_sm", output_dir="/kaggle/working/custom_model", n_iter=100)
# nlp = spacy.load('/kaggle/working/custom_model') #loading the custom model into the same variable
# Test again

# tr4w = TextRankForKeyword()

# tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)

# tr4w.get_keywords(10)
#TODO: Code against Catastrophic Forgetting: https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
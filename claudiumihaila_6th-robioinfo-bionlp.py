import codecs



with codecs.open("/kaggle/input/geniaevent2011/dev/PMC-1134658-00-TIAB.txt", "r", "utf-8") as fin:

    print(fin.readlines())
with codecs.open("/kaggle/input/geniaevent2011/dev/PMC-1134658-00-TIAB.a1", "r", "utf-8") as fin:

    print(fin.readlines())
with codecs.open("/kaggle/input/geniaevent2011/dev/PMC-1134658-00-TIAB.a2", "r", "utf-8") as fin:

    print(fin.readlines())
import os



def read_input_files(subset):

    texts = {}

    ents = {}

    eves = {}

    ent_labels = set()

    eve_labels = set()

    for dirname, _, filenames in os.walk('/kaggle/input/geniaevent2011/' + subset):

        for filename in filenames:

            fn = os.path.join(dirname, filename)

            with codecs.open(fn, 'r', 'utf8') as fin:

                # process original text files (.txt)

                if filename.endswith('.txt'):

                    texts[filename[:-4]] = "".join(fin.readlines())

                # process entity files (.a1)

                elif filename.endswith('.a1'):

                    if filename[:-3] not in ents:

                        ents[filename[:-3]] = {}

                    for l in fin.readlines():

                        if l.strip() != '':

                            tabs = l.split('\t')

                            spaces = tabs[1].split(' ')

                            ents[filename[:-3]][tabs[0]] = (int(spaces[1]), int(spaces[2]), spaces[0])

                            ent_labels.add(spaces[0])

                # process event files (.a2)

                elif filename.endswith('.a2'):

                    if filename[:-3] not in eves:

                        eves[filename[:-3]] = {}

                    if filename[:-3] not in ents:

                        ents[filename[:-3]] = {}

                    for l in fin.readlines():

                        if l.strip() != '':

                            l = l.strip()

                            if l[0] == "T":

                                # entity

                                tabs = l.split('\t')

                                spaces = tabs[1].split(' ')

                                ents[filename[:-3]][tabs[0]] = (int(spaces[1]), int(spaces[2]), spaces[0])

                                ent_labels.add(spaces[0])

                            elif l[0] == "E":

                                # event

                                tabs = l.split('\t')

                                eves[filename[:-3]][tabs[0]] = tabs[1].split(' ')

                    

    return texts, ents, eves, list(ent_labels), list(eve_labels)

# Any results you write to the current directory are saved as output.
texts, ents, eves, ent_labels, eve_labels = read_input_files('dev/')

# print(ent_labels)

# print(ents)
TRAIN_DATA = [(texts[k], {"entities": list(ents[k].values()) if k in ents else []}) for k in texts]

# print(TRAIN_DATA)
import spacy

import en_core_web_sm

from spacy import displacy



nlp = en_core_web_sm.load()

texts = ["This is a text.", "And now Andy will process it with Spacy. Great things are happening in lovely Cluj Napoca."]

for doc in nlp.pipe(texts):

    print(list(doc.sents))

#     print([[(token.text, token.lemma_, token.pos_, token.tag_) for token in sent] for sent in doc.sents])

#     print(list(doc.noun_chunks))

#     print([(ent.text, ent.label_) for ent in doc.ents])

#     displacy.render(doc.sents, style="dep")

#     displacy.render(doc.sents, style="ent")
from __future__ import unicode_literals, print_function



import random

from pathlib import Path

import spacy

from spacy.util import minibatch, compounding

def train(train_data, labels):

    random.seed(0)

    # start off with a blank model, for the English language, and extract the NER pipe so it can be set up

    nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:

        ner = nlp.create_pipe("ner")

        nlp.add_pipe(ner)

    # otherwise, get it

    else:

        ner = nlp.get_pipe("ner")



    # add new entity labels to the entity recogniser

    [ner.add_label(label) for label in labels] 

    n_iter = 10 # how many times should repeat the training procedure

    

    optimiser = nlp.begin_training()

    move_names = list(ner.move_names)

    print(move_names)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

    

    with nlp.disable_pipes(*other_pipes):  # only train NER, ignore the other pipes

        sizes = compounding(1.0, 4.0, 1.001)

        # batch up the examples using spaCy's minibatch

        for itn in range(n_iter):

            random.shuffle(train_data)

            batches = minibatch(train_data, size=sizes)

            losses = {}

            for batch in batches:

                texts, annotations = zip(*batch)

                nlp.update(texts, annotations, sgd=optimiser, drop=0.35, losses=losses)

            print("Losses", losses)

    return nlp
model = train(TRAIN_DATA, ent_labels)
# test the trained model

def test(model, text):

    doc = model(text)

    print("Entities in '%s'" % text)

    for ent in doc.ents:

        print(ent.label_, ent.text)

    displacy.render(doc, style="ent")
test_text = "MTb induces NFAT5 gene expression via the MyD88-dependent signaling cascade."

test(model, test_text)
!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_sm-0.2.0.tar.gz
import scispacy

import spacy

import en_core_sci_sm

from scispacy.umls_linking import UmlsEntityLinker

from scispacy.abbreviation import AbbreviationDetector
nlp = en_core_sci_sm.load()

# Add the abbreviation pipe to the spacy pipeline.

abbreviation_pipe = AbbreviationDetector(nlp)

linker = UmlsEntityLinker(resolve_abbreviations=True)

nlp.add_pipe(abbreviation_pipe)

nlp.add_pipe(linker)

text = """

Myeloid-derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. 

They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC).

MTb induces NFAT5 gene expression via the MyD88-dependent signaling cascade.

"""

doc = nlp(text)



print(list(doc.sents))

print(doc.ents)
for abrv in doc._.abbreviations:

    print(f"{abrv} \t ({abrv.start}, {abrv.end}) \t {abrv._.long_form}")
for entity in doc.ents:

    if len(entity._.umls_ents) > 0:

        print(entity, linker.umls.cui_to_entity[entity._.umls_ents[0][0]], "\n\n\n\n")

#     for umls_ent in entity._.umls_ents:

#         print(entity, linker.umls.cui_to_entity[umls_ent[0]], "\n\n\n\n")
displacy.render(doc, style="ent")

displacy.render(doc, style="dep")
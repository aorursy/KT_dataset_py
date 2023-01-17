import json

import os

import time



import spacy

from tqdm import tqdm

from unidecode import unidecode



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
NUM_KLAT_LINES = 5_343_564  # link_annotated_text.jsonl

MAX_PAGES = 1_000_000

MIN_TOKENS_IN_SENTENCE = 5

BATCH_SIZE = 200
nlp = spacy.load('en_core_web_sm', deactivate=['tagger', 'parser', 'ner'])
in_fname = '/kaggle/input/kensho-derived-wikimedia-data/link_annotated_text.jsonl'

corpus_fname = 'wikipedia_intros_sentences.txt'

with open(in_fname, 'r') as ifp, open(corpus_fname, 'w') as ofp:

    batch = []

    for iline, line in enumerate(tqdm(ifp, total=NUM_KLAT_LINES, desc='writing sentences')):

        if iline >= MAX_PAGES:

            break

        page = json.loads(line)

        text = unidecode(page['sections'][0]['text'])

        batch.append(text)

        if len(batch) >= BATCH_SIZE:

            docs = nlp.pipe(batch)

            for doc in docs:

                sentences = [sent.text for sent in doc.sents if len(sent) > MIN_TOKENS_IN_SENTENCE]

                for sentence in sentences:

                    ofp.write('{}\n'.format(sentence))

            batch = []

            

    if len(batch) >= BATCH_SIZE:

        docs = nlp.pipe(batch)

        for doc in docs:

            sentences = [sent.text for sent in doc.sents if len(sent) > MIN_TOKENS_IN_SENTENCE]

            for sentence in sentences:

                ofp.write('{}\n'.format(sentence))
!head wikipedia_intros_sentences.txt
!ls -lh
import numpy as np

import pandas as pd

import spacy

from spacy import displacy

nlp = spacy.load('en_core_web_sm')

import nltk

from sklearn import *



test = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t').rename(columns={'A': 'A_Noun', 'B': 'B_Noun'})

sub = pd.read_csv('../input/sample_submission_stage_1.csv')

test.shape, sub.shape
test['mask'] = test['Text'].map(lambda x: '*' * len(x))

test['mask'] = test.apply(lambda r: r['mask'][: r['Pronoun-offset']] + 'P' * len(str(r['Pronoun'])) + r['mask'][r['Pronoun-offset'] + len(str(r['Pronoun'])): ], axis=1)

test['mask'] = test.apply(lambda r: r['mask'][: r['A-offset']] + 'A' * len(str(r['A_Noun'])) + r['mask'][r['A-offset'] + len(str(r['A_Noun'])): ], axis=1)

test['mask'] = test.apply(lambda r: r['mask'][: r['B-offset']] + 'B' * len(str(r['B_Noun'])) + r['mask'][r['B-offset'] + len(str(r['B_Noun'])): ], axis=1)

test['section_min'] = test[['Pronoun-offset', 'A-offset', 'B-offset']].min(axis=1)

test['Pronoun-offset2'] = test['Pronoun-offset'] + test['Pronoun'].map(len)

test['A-offset2'] = test['A-offset'] + test['A_Noun'].map(len)

test['B-offset2'] = test['B-offset'] + test['B_Noun'].map(len)                               

test['section_max'] = test[['Pronoun-offset2', 'A-offset2', 'B-offset2']].max(axis=1)

print(test['Text'][0][test['section_min'][0]:test['section_max'][0]])

print(test['mask'][0][test['section_min'][0]:test['section_max'][0]])
doc = nlp(test['Text'][0][test['section_min'][0]:test['section_max'][0]])

tokens = pd.DataFrame(

    [[token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop, [child for child in token.children]] for token in doc],

    columns=['text', 'lemma', 'pos', 'tag', 'dep', 'shape', 'is_alpha', 'is_stop', 'child'])

tokens
test['A'] = test.apply(lambda r: 0.45 if r['A-offset'] == r['section_min'] else 0.4, axis=1)

test['B'] = test.apply(lambda r: 0.3 if r['B-offset'] == r['section_min'] else 0.45, axis=1)

test['NEITHER'] = 1.0 - (test['A'] + test['B'])
#metrics.log_loss()
test[['ID', 'A', 'B', 'NEITHER']].to_csv('submission.csv', index=False)
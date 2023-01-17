import numpy as np

import pandas as pd
gap_train = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-development.tsv", delimiter='\t')

gap_test = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv", delimiter='\t')

gap_valid = pd.read_csv("https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-validation.tsv", delimiter='\t')
test_stage_1 = pd.read_csv('../input/test_stage_1.tsv', delimiter='\t')

sub = pd.read_csv('../input/sample_submission_stage_1.csv')
# !pip install https://github.com/huggingface/neuralcoref-models/releases/download/en_coref_md-3.0.0/en_coref_md-3.0.0.tar.gz

# !pip install cymem==1.31.2 spacy==2.0.12
# import en_coref_md

# from spacy.tokens import Doc
# nlp = en_coref_md.load()
# class WhitespaceTokenizer(object):

#     def __init__(self, vocab):

#         self.vocab = vocab

#     def __call__(self, text):

#         words = text.split(' ')

#         words = [word for word in words]

#         spaces = [True] * len(words)

#         return Doc(self.vocab, words=words, spaces=spaces)

# nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
# gap_train.iloc[0:2, :]
# def check_coref(row):

#     text = row["Text"]

#     words = text.split()



#     pronoun = row["Pronoun"]

#     pronoun_off = row["Pronoun-offset"]

#     A = row["A"]

#     len_a = len(A.split())

#     A_off = row["A-offset"]

#     B = row["B"]

#     len_b = len(B.split())

#     B_off = row["B-offset"]

#     position = 0

#     for i, word in enumerate(words):

#         if position == pronoun_off:

#             pronoun_word_index = i

#         if position == A_off:

#             A_off_word_index = (i, i+len_a)

#         if position == B_off:

#             B_off_word_index = (i, i+len_b)

#         position += len(word) + 1

#     #print(A_off_word_index, B_off_word_index, pronoun_word_index)

#     doc = nlp(text)

#     token = None

#     try:

#         token = doc[pronoun_word_index]

#     except:

#         print(pronoun, pronoun_off)

#     try:

#         print(token, A, B, token._.coref_clusters)

#     except:

#         return [0, 0, 1]

    

# gap_train.apply(check_coref, axis=1)
# test_stage_1
def measure_dist(row):

    pro_off = row["Pronoun-offset"]

    a_off = row["A-offset"]

    b_off = row["B-offset"]

    a_dist = np.abs(pro_off - a_off)

    b_dist = np.abs(pro_off - b_off)

    dist_tot = a_dist + b_dist

    a_val = a_dist/dist_tot

    b_val = b_dist/dist_tot

    neither = .5

    return [a_val, b_val, neither]

test_stage_1["preds"] = test_stage_1.apply(measure_dist, axis = 1)
test = test_stage_1.preds.apply(pd.Series)
sub["A"] = test[0]

sub["B"] = test[1]

sub["NEITHER"] = test[2]
sub[['ID', 'A', 'B', 'NEITHER']].to_csv('submission.csv', index=False)
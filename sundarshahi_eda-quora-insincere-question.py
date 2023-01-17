from gensim.utils import simple_preprocess
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import spacy

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.target.hist(bins=3)

train_df[['target']].boxplot()
train_df.groupby('target').count()
nlp = spacy.load('en', disable=['ner'])

negative_questions = train_df.loc[train_df.target == 1].question_text
all_tokens = [token
              for question in negative_questions
              for token in nlp(question)]
clean_words = [token.lemma_
               for token in all_tokens
               if not token.is_punct
               and token.lemma_ != '-PRON-'
               and not nlp.vocab[token.lemma_].is_stop]
clean_word_dist = nltk.FreqDist(clean_words)
clean_word_dist.most_common(30)
puncts = [token.text for token in all_tokens if token.is_punct]
punct_dist = nltk.FreqDist(puncts)
punct_dist.most_common(10)
subjects = [token.lemma_
            for token in all_tokens
            if not token.is_stop
            and 'subj' in token.dep_ and 'subj' in token.dep_
            and token.pos_ in {'PROPN', 'NOUN'}]
subject_dist = nltk.FreqDist(subjects)
subject_dist.most_common(20)
train_lengths = [len(simple_preprocess(doc))
                 for doc in train_df.question_text]
test_lengths = [len(simple_preprocess(doc))
                for doc in test_df.question_text]
combined_lengths = train_lengths + test_lengths
pd.Series(combined_lengths).describe(percentiles=[.25, .5, .75, .9, .95, .99])

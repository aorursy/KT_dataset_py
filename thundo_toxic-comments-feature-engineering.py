import pandas as pd
import collections
import string
import seaborn as sns
import matplotlib.pyplot as plt

from nltk import pos_tag
from nltk.corpus import stopwords
COMMENT = 'comment_text'
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
train = pd.read_csv("../input/train.csv", encoding="utf-8")
print(train.head())
print("Train: %s samples" % len(train))
train['total_length'] = train[COMMENT].apply(len)

train['words'] = train[COMMENT].apply(lambda comment: len(comment.split()))
train['words_vs_length'] = train['words'] / train['total_length']

train['capitals'] = train[COMMENT].apply(lambda comment: sum(1 for c in comment if c.isupper()))
train['capitals_vs_length'] = train['capitals'] / train['total_length']
train['capitals_vs_words'] = train['capitals'] / train['words']

train['paragraphs'] = train[COMMENT].apply(lambda comment: comment.count('\n'))
train['paragraphs_vs_length'] = train['paragraphs'] / train['total_length']
train['paragraphs_vs_words'] = train['paragraphs'] / train['words']

eng_stopwords = set(stopwords.words("english"))
train['stopwords'] = train[COMMENT].apply(lambda comment: sum(comment.count(w) for w in eng_stopwords))
train['stopwords_vs_length'] = train['stopwords'] / train['total_length']
train['stopwords_vs_words'] = train['stopwords'] / train['words']

train['exclamation_marks'] = train[COMMENT].apply(lambda comment: comment.count('!'))
train['exclamation_marks_vs_length'] = train['exclamation_marks'] / train['total_length']
train['exclamation_marks_vs_words'] = train['exclamation_marks'] / train['words']

train['question_marks'] = train[COMMENT].apply(lambda comment: comment.count('?'))
train['question_marks_vs_length'] = train['question_marks'] / train['total_length']
train['question_marks_vs_words'] = train['question_marks'] / train['words']

train['punctuation'] = train[COMMENT].apply(
    lambda comment: sum(comment.count(w) for w in string.punctuation))
train['punctuation_vs_length'] = train['punctuation'] / train['total_length']
train['punctuation_vs_words'] = train['punctuation'] / train['words']

train['unique_words'] = train[COMMENT].apply(
    lambda comment: len(set(w for w in comment.split())))
train['unique_words_vs_length'] = train['unique_words'] / train['total_length']
train['unique_words_vs_words'] = train['unique_words'] / train['words']

repeated_threshold = 15
def count_repeated(text):
    text_splitted = text.split()
    word_counts = collections.Counter(text_splitted)
    return sum(count for word, count in sorted(word_counts.items()) if count > repeated_threshold)

train['repeated_words'] = train[COMMENT].apply(lambda comment: count_repeated(comment))
train['repeated_words_vs_length'] = train['repeated_words'] / train['total_length']
train['repeated_words_vs_words'] = train['repeated_words'] / train['words']

train['mentions'] = train[COMMENT].apply(
    lambda comment: comment.count("User:"))
train['mentions_vs_length'] = train['mentions'] / train['total_length']
train['mentions_vs_words'] = train['mentions'] / train['words']


train['smilies'] = train[COMMENT].apply(
    lambda comment: sum(comment.count(w) for w in (':-)', ':)', ';-)', ';)')))
train['smilies_vs_length'] = train['smilies'] / train['total_length']
train['smilies_vs_words'] = train['smilies'] / train['words']

train['symbols'] = train[COMMENT].apply(
    lambda comment: sum(comment.count(w) for w in '*&#$%“”¨«»®´·º½¾¿¡§£₤‘’'))
train['symbols_vs_length'] = train['symbols'] / train['total_length']
train['symbols_vs_words'] = train['symbols'] / train['words']
def tag_part_of_speech(text):
    text_splited = text.split(' ')
    text_splited = [''.join(c for c in s if c not in string.punctuation) for s in text_splited]
    text_splited = [s for s in text_splited if s]
    pos_list = pos_tag(text_splited)
    noun_count = len([w for w in pos_list if w[1] in ('NN','NNP','NNPS','NNS')])
    adjective_count = len([w for w in pos_list if w[1] in ('JJ','JJR','JJS')])
    verb_count = len([w for w in pos_list if w[1] in ('VB','VBD','VBG','VBN','VBP','VBZ')])
    return[noun_count, adjective_count, verb_count]


train['nouns'], train['adjectives'], train['verbs'] = zip(*train[COMMENT].apply(
    lambda comment: tag_part_of_speech(comment)))
train['nouns_vs_length'] = train['nouns'] / train['total_length']
train['adjectives_vs_length'] = train['adjectives'] / train['total_length']
train['verbs_vs_length'] = train['verbs'] / train['total_length']
train['nouns_vs_words'] = train['nouns'] / train['words']
train['adjectives_vs_words'] = train['adjectives'] / train['words']
train['verbs_vs_words'] = train['verbs'] / train['words']
train.head()
features = ('total_length', 
            'words', 'words_vs_length',
            'capitals', 'capitals_vs_length', 'capitals_vs_words',
            'paragraphs', 'paragraphs_vs_length', 'paragraphs_vs_words',
            'stopwords', 'stopwords_vs_length', 'stopwords_vs_words',
            'exclamation_marks', 'exclamation_marks_vs_length', 'exclamation_marks_vs_words',
            'question_marks', 'question_marks_vs_length', 'question_marks_vs_words',
            'punctuation', 'punctuation_vs_length', 'punctuation_vs_words',
            'unique_words', 'unique_words_vs_length', 'unique_words_vs_words',
            'repeated_words', 'repeated_words_vs_length', 'repeated_words_vs_words',
            'mentions', 'mentions_vs_words', 'mentions_vs_length',
            'smilies', 'smilies_vs_length', 'smilies_vs_words',
            'symbols', 'symbols_vs_length', 'symbols_vs_words',
            'nouns', 'nouns_vs_words', 'nouns_vs_length', 
            'adjectives', 'adjectives_vs_words', 'adjectives_vs_length',
            'verbs', 'verbs_vs_words', 'verbs_vs_length',
           )
train['none'] = 1 - train[LABELS].max(axis=1)
columns = LABELS + ['none']

rows = [{c:train[f].corr(train[c]) for c in columns} for f in features]
train_correlations = pd.DataFrame(rows, index=features)
train_correlations
import seaborn as sns

fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(train_correlations, annot=True, vmin=-0.23, vmax=0.23, center=0.0, ax=ax)
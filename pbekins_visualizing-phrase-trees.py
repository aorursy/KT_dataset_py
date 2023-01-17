import pandas as pd 



train_file = '../input/sentiment-analysis-on-movie-reviews/train.tsv.zip'

test_file = '../input/sentiment-analysis-on-movie-reviews/test.tsv.zip'



train = pd.read_csv(train_file, delimiter = '\t', compression = 'zip')

test = pd.read_csv(test_file, delimiter = '\t', compression = 'zip')
train.shape
test.shape
import nltk

from nltk import RegexpTokenizer

def n_grams(phrase):

    tokenizer = nltk.RegexpTokenizer(r"\w+")

    words = tokenizer.tokenize(phrase)

    return len(words)



train['N'] = train['Phrase'].apply(n_grams)

test['N'] = test['Phrase'].apply(n_grams)
train.head()
train['N'].hist(bins = 20)

train['N'].max()
test['N'].hist(bins = 20)

test['N'].max()
train_sentences = train.groupby(['SentenceId']).first().reset_index()

train_sentences['N'].hist(bins = 20)

train_sentences.shape
test_sentences = test.groupby(['SentenceId']).first().reset_index()

test_sentences['N'].hist(bins = 20)
from nltk import Tree

sent =  "(S (NP (A Poor ) (N John)) (VP (V ran ) (Adv away)))"

tree = Tree.fromstring(sent)

tree.pretty_print()
train.loc[(train['SentenceId'] == 2)]
phrases = train.loc[(train['SentenceId'] == 2)]['Phrase'].to_list()

sentiments = train.loc[(train['SentenceId'] == 2)]['Sentiment'].to_list()

root = phrases[0]

for p, s in zip(phrases,sentiments):

    start = root.index(p)

    end = start + len(p) + len(str(s)) + 2

    root = root[:start] + '(' + str(s) + ' ' + root[start:]

    root = root[:end] + ')' + root[end:]

    print(root)
tree = Tree.fromstring(root)

tree.pretty_print()
def phrase_tree(phrase_group):

    phrases = phrase_group['Phrase'].to_list()

    sentiments = phrase_group['Sentiment'].to_list()

    root = phrases[0]

    for p, s in zip(phrases,sentiments):

        try:

            start = root.index(p)

        except:

            root = 'error'

        else:

            end = start + len(p) + len(str(s)) + 2

            root = root[:start] + '(' + str(s) + ' ' + root[start:]

            root = root[:end] + ')' + root[end:]

    return root



train_trees = []

train_groups = train.groupby(['SentenceId'])

for key, group in train_groups:

    root = phrase_tree(group)

    train_trees.append((key,root))
errors = 0

for tree in train_trees:

    if tree[1] == 'error':

        errors += 1

print(errors)
train.loc[(train['SentenceId'] == 8382)]
!ls ../input/aclimdb/aclImdb
import os

import random

import re

import spacy

import time

from tqdm import tqdm



random.seed(2)  # 确保每次运行结果一致
with open('../input/aclimdb/aclImdb/test/pos/4_10.txt', 'r', encoding='utf8') as f:

    text = f.read()

print(type(text))

print(text)
text = re.sub(r'\s*<br /><br />\s*', ' ', text)

print(text)
def read_imdb(data_path, folder):

    """

    @param data_path: 

    @param folder: 

    @return: 

    """

    result = []

    for label in ['pos', 'neg']:

        folder_path = (os.path.join(data_path, folder, label))

        for file in tqdm(os.listdir(folder_path)):

            file_path = os.path.join(folder_path, file)

            with open(file_path, 'r', encoding='utf8') as f:

                text = f.read()

                text = re.sub(r'\s*<br /><br />\s*', ' ', text)

            if label == 'pos':

                label_ = 1

            else:

                label_ = 0

            result.append([text, label_])

    random.shuffle(result)

    return result
data_path = '../input/aclimdb/aclImdb'

folder = 'train'

label = 'pos'

folder_path = os.path.join(data_path, folder, label)  # 将所有参数拼接起来形成一个路径并返回

print(folder_path)

print(type(folder_path))
for i in tqdm(range(5)):

    time.sleep(1)
files = os.listdir(folder_path)  # 返回给定路径下的所有文件的文件名

print(type(files))

print(len(files))

print(files[: 10])
data_path = '../input/aclimdb/aclImdb'

test_data = read_imdb(data_path, 'test')
def evaluate(labels, predictions):

    """

    @param labels: 

    @param predictions: 

    @return: 

    """

    length = len(labels)

    TP, FP, FN = 0, 0, 0

    for i in range(length):

        y, y_hat = labels[i], predictions[i]

        TP += (y == 1 and y_hat == 1)

        FP += (y == 0 and y_hat == 1)

        FN += (y == 1 and y_hat == 0)

    P = TP / (TP + FP)

    R = TP / (TP + FN)

    F1 = 2 * P * R / (P + R)

    return P, R, F1
labels = [0, 0, 1, 1, 1]

predictions = [0, 1, 0, 1, 1]

P, R, F1 = evaluate(labels, predictions)

print('%.2f%%, %.2f%%, %.2f%%' % (P * 100, R * 100, F1 * 100))
def read_senti_dict(data_path, filename):

    """

    @param data_path: 情感词典文件目录

    @param filename: 情感词典文件名

    @return: 由情感词典文件中所有情感词构成的set

    """

    with open(os.path.join(data_path, filename), 'r') as f:

        words = [word.strip().lower() for word in f]

    return set(words)
!ls ../input/sentimentdictionary/
sentiment_dict_path = '../input/sentimentdictionary/Hu and Liu Sentiment Lexicon'

pos_file, neg_file = 'positive-words.txt', 'negative-words.txt'

pos_words = read_senti_dict(sentiment_dict_path, pos_file)

neg_words = read_senti_dict(sentiment_dict_path, neg_file)
print(list(pos_words)[: 10])

print(list(neg_words)[: 10])

print(len(pos_words))

print(len(neg_words))
# source: https://spacy.io/usage/linguistic-features

nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")



for token in doc:

    print("%s" % token.text)
def predict(test_data, pos_words, neg_words):

    """

    @param test_data: 

    @param pos_words: 

    @param neg_words: 

    @return: 

    """

    nlp = spacy.load('en_core_web_sm')

    result = []

    for text, _ in tqdm(test_data):

        doc = nlp(text)

        score = 0

        for token in doc:

            score += (token.text.lower() in pos_words)

            score -= (token.text.lower() in neg_words)

        result.append(int(score > 0))

    return result
predictions = predict(test_data[: 2000], pos_words, neg_words)
test_labels = [label for _, label in test_data[: 2000]]

precision, recall, F1 = evaluate(test_labels, predictions)

print('precision: %.1f%%' % (precision * 100))

print('recall: %.1f%%' % (recall * 100))

print('F1: %.1f%%' % (F1 * 100))
# source: https://spacy.io/usage/linguistic-features

nlp = spacy.load("en_core_web_sm")

doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")



for token in doc:

    print("%-10s%-10s%-10s%-10s" % (token.text, token.lemma_, token.pos_, token.is_punct))
# source: https://spacy.io/usage/linguistic-features

doc = nlp("This is a sentence. This is another sentence.")

for sent in doc.sents:

    print(sent.text)

    for token in sent:

        print("%-10s%-10s%-10s%-10s" % (token.text, token.lemma_, token.pos_, token.is_punct))

    print()
def predict_rule_based(test_data, pos_words, neg_words):

    """

    @param test_data: 

    @param pos_words: 

    @param neg_words: 

    @return: 

    """

    nlp = spacy.load('en_core_web_sm')

    result = []

    for text, _ in tqdm(test_data):

        doc = nlp(text)

        score = 0

        for sent in doc.sents:

            sent_score = 0

            subsent_score = 0

            neg_score = 1

            for token in sent:

                subsent_score += (token.text.lower() in pos_words)

                subsent_score -= (token.text.lower() in neg_words)

                if token.lemma_.lower() in ('not', 'no'):

                    neg_score *= -1

                if token.is_punct:

                    sent_score += subsent_score * neg_score

                    subsent_score = 0

                    neg_score = 1

            score += sent_score

        result.append(int(score > 0))

    return result
predictions = predict_rule_based(test_data[: 2000], pos_words, neg_words)

test_labels = [label for _, label in test_data[: 2000]]

precision, recall, F1 = evaluate(test_labels, predictions)

print('precision: %.1f%%' % (precision * 100))

print('recall: %.1f%%' % (recall * 100))

print('F1: %.1f%%' % (F1 * 100))
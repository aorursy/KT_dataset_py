# Here we will show how 5-gram language model works:

import random

wordList = ['here', 'there', 'today', 'he', 'go']

prefixSentence = 'Can you please come '

totalSentences = 10000
sentenceList = []

for i in range(totalSentences):

    sentence = prefixSentence + wordList[random.randint(0, 4)]

    sentenceList.append(sentence)
print(sentenceList)
cntList = [0, 0, 0, 0, 0]

for i in sentenceList:

    for j in range(5):

        if wordList[j] in i:

            cntList[j] = cntList[j] + 1

print(cntList)
bigIdx = 0

curBig = -1

for i in range(5):

    if curBig < cntList[i]:

        bigIdx = i

        curBig = cntList[i]

print(curBig, ' ', bigIdx)
print(prefixSentence + wordList[bigIdx])
# clean data for poems
import json
poems = []

with open("../input/poem.json","r") as load_file:

    poems = json.load(load_file)

    print(poems[42])

    print(type(poems[42]))
print(type(poems[42]["paragraphs"]))
print(type(poems))
poems_processed = []

for poem in poems:

    for sentence in poem['paragraphs']:

        if '，' not in sentence:

            continue

        if not len(sentence) == 12:

            continue

        s1, s2 = sentence.split('，')

        poems_processed.append(s1)

        poems_processed.append(s2[:-1])
print(poems_processed[:20])
import jieba 

import itertools

from collections import Counter
text = '接下来，莫沃维奇和关一帆又发现了一件令他们激动的事情：他们能看到星空，在各个方向上都能看到。他们清楚地看见，在宇宙的永恒之夜中，银河系在灿烂地延伸着。他们知道自己此时仍身处飞船中，三人都没有穿宇宙服，都在呼吸着飞船中的空气，但在第四个维度上，他们暴露在太空中。作为宇航员，三个人都曾经历过无数次太空行走，但从未感觉到自己在太空中暴露得这样彻底。以往太空行走时，他们至少包裹在宇宙服中，而现在，没有任何东西挡在他们和宇宙之间，周围这展现出无限细节的飞船对星空没有丝毫遮挡，在第四维度上，整个宇宙与飞船也是并列的。'

print(text)
text_cut = jieba.cut(text)

text_cut = ' '.join(text_cut)

print(text_cut)
l = ['1', '233', '666', '4', '5']

print(' '.join(l))
jieba.suggest_freq("莫沃维奇", True)
text_cut = jieba.cut(text)

text_cut = '/'.join(text_cut)

print(text_cut)

print(type(text_cut))
def build_vocab(text, vocab_lim):

    word_cnt = Counter(text)

    vocab_inv = [x[0] for x in word_cnt.most_common(vocab_lim)] # 

    vocab_inv = list(sorted(vocab_inv))

    vocab = {x: index for index, x in enumerate(vocab_inv)}

    return vocab, vocab_inv
text_cut_list = text_cut.split('/')

print(text_cut_list)
vocab, vocab_inv = build_vocab(text_cut_list, 4000)
print(vocab)
num_text_cut = [vocab[word] for word in text_cut_list]
print(num_text_cut)
import numpy as np

import pandas as pd

import os

import warnings

warnings.filterwarnings("ignore")



from nltk import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from collections import defaultdict

from nltk.sentiment.vader import SentimentIntensityAnalyzer
train_orig = pd.read_csv("../input/quora-question-pairs-data-cleaning/train_orig.csv")

test_orig = pd.read_csv("../input/quora-question-pairs-data-cleaning/test_orig.csv")

train_stop = pd.read_csv("../input/quora-question-pairs-data-cleaning/train_stop.csv")

test_stop = pd.read_csv("../input/quora-question-pairs-data-cleaning/test_stop.csv")

train_stem = pd.read_csv("../input/quora-question-pairs-data-cleaning/train_stem.csv")

test_stem = pd.read_csv("../input/quora-question-pairs-data-cleaning/test_stem.csv")

train_lem = pd.read_csv("../input/quora-question-pairs-data-cleaning/train_lem.csv")

test_lem = pd.read_csv("../input/quora-question-pairs-data-cleaning/test_lem.csv")
train_orig = train_orig.fillna("")

test_orig = test_orig.fillna("")

train_stop = train_stop.fillna("")

test_stop = test_stop.fillna("")

train_stem = train_stem.fillna("")

test_stem = test_stem.fillna("")

train_lem = train_lem.fillna("")

test_lem = test_lem.fillna("")
train = pd.DataFrame(index = train_orig.index)

test = pd.DataFrame(index = test_orig.index)

trainlabel = train_orig[["is_duplicate"]]
len1 = train_orig["question1"].apply(len)

len2 = train_orig["question2"].apply(len)

train["diff_char"] = abs(len1 - len2)

train["diff_char_rate"] = 2 * abs(len1 - len2) / (len1 + len2)



len3 = test_orig["question1"].apply(len)

len4 = test_orig["question2"].apply(len)

test["diff_char"] = abs(len3 - len4)

test["diff_char_rate"] = 2 * abs(len3 - len4) / (len3 + len4)



del len1, len2, len3, len4
def words_count(text):

    wordlist = word_tokenize(text)

    count = len(wordlist)

    return count



count1 = train_orig["question1"].apply(words_count)

count2 = train_orig["question2"].apply(words_count)

train["diff_words"] = abs(count1 - count2)

train["diff_words_rate"] = 2 * abs(count1 - count2) / (count1 + count2)



count3 = test_orig["question1"].apply(words_count)

count4 = test_orig["question2"].apply(words_count)

test["diff_words"] = abs(count3 - count4)

test["diff_words_rate"] = 2 * abs(count3 - count4) / (count3 + count4)
def shared_words_count(text1, text2):

    wordlist1 = word_tokenize(text1)

    wordlist2 = word_tokenize(text2)

    wordset1 = set(wordlist1)

    wordset2 = set(wordlist2)

    inter = wordset1 & wordset2

    union = wordset1 | wordset2

    count = len(inter)

    rate = 2 * count / (len(wordset1) + len(wordset2) + 1)  # 为了防止wordset1和wordset2同时为空，也即text1和text2都是只包含一个空格的字符串

    jaccard = count / (len(union) + 1)

    return pd.Series([count, rate, jaccard])
share_train_orig = train_orig[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_train_stop = train_stop[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_train_stem = train_stem[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_train_lem = train_lem[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_words_train = pd.concat([share_train_orig, share_train_stop, share_train_stem, share_train_lem], axis = 1)

share_words_train.columns = ["share_words_count_orig", "share_words_rate_orig", "jaccard_orig", 

                             "share_words_count_stop", "share_words_rate_stop", "jaccard_stop", 

                             "share_words_count_stem", "share_words_rate_stem", "jaccard_stem",

                             "share_words_count_lem", "share_words_rate_lem", "jaccard_lem"]

train = pd.concat([train, share_words_train], axis = 1)



share_test_orig = test_orig[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_test_stop = test_stop[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_test_stem = test_stem[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_test_lem = test_lem[["question1", "question2"]].apply(lambda x: shared_words_count(x[0], x[1]), axis = 1)

share_words_test = pd.concat([share_test_orig, share_test_stop, share_test_stem, share_test_lem], axis = 1)

share_words_test.columns = ["share_words_count_orig", "share_words_rate_orig", "jaccard_orig", 

                            "share_words_count_stop", "share_words_rate_stop", "jaccard_stop", 

                            "share_words_count_stem", "share_words_rate_stem", "jaccard_stem",

                            "share_words_count_lem", "share_words_rate_lem", "jaccard_lem"]

test = pd.concat([test, share_words_test], axis = 1)



del share_train_orig, share_train_stop, share_train_stem, share_train_lem, share_words_train

del share_test_orig, share_test_stop, share_test_stem, share_test_lem, share_words_test
wordbag = pd.concat([train_orig["question1"], train_orig["question2"]], axis = 0)

tfidf = TfidfVectorizer(analyzer = "word", stop_words = "english", lowercase = True)

tfidf.fit(wordbag)



del wordbag
tfidf_q1_train = tfidf.transform(train_orig["question1"])

tfidf_q2_train = tfidf.transform(train_orig["question2"])



diff = tfidf_q1_train - tfidf_q2_train

diff_tfidf_L1_train = np.sum(np.abs(diff), axis = 1)  # 统一用numpy的函数比较好

diff_tfidf_L2_train = np.sum(diff.multiply(diff), axis = 1)

diff_tfidf_L1_norm_train = 2 * np.array(np.sum(np.abs(diff), axis = 1)) / pd.DataFrame(count1 + count2).values

diff_tfidf_L2_norm_train = 2 * np.array(np.sum(diff.multiply(diff), axis = 1)) / pd.DataFrame(count1 + count2).values

# tfidf_q1_train和tfidf_q2_train，以及diff都是稀疏矩阵

# 转换成数组再做对应元素的运算将会报错，可以用matrix对象自带的方法multiply实现

cos_tfidf_train = np.sum(tfidf_q1_train.multiply(tfidf_q2_train), axis = 1)  # 由于词的tfidf表示是经过标准化的，所以内积即为夹角余弦值



train["diff_tfidf_L1"] = diff_tfidf_L1_train

train["diff_tfidf_L2"] = diff_tfidf_L2_train

train["diff_tfidf_L1_norm"] = diff_tfidf_L1_norm_train

train["diff_tfidf_L2_norm"] = diff_tfidf_L2_norm_train

train["cos_tfidf"] = cos_tfidf_train



del tfidf_q1_train, tfidf_q2_train, diff, diff_tfidf_L1_train, diff_tfidf_L2_train

del diff_tfidf_L1_norm_train, diff_tfidf_L2_norm_train, cos_tfidf_train
tfidf_q1_test = tfidf.transform(test_orig["question1"])

tfidf_q2_test = tfidf.transform(test_orig["question2"])



diff = tfidf_q1_test - tfidf_q2_test

diff_tfidf_L1_test = np.sum(np.abs(diff), axis = 1)

diff_tfidf_L2_test = np.sum(diff.multiply(diff), axis = 1)

diff_tfidf_L1_norm_test = 2 * np.array(np.sum(np.abs(diff), axis = 1)) / pd.DataFrame(count3 + count4).values

diff_tfidf_L2_norm_test = 2 * np.array(np.sum(diff.multiply(diff), axis = 1)) / pd.DataFrame(count3 + count4).values

cos_tfidf_test = np.sum(tfidf_q1_test.multiply(tfidf_q2_test), axis = 1)



test["diff_tfidf_L1"] = diff_tfidf_L1_test

test["diff_tfidf_L2"] = diff_tfidf_L2_test

test["diff_tfidf_L1_norm"] = diff_tfidf_L1_norm_test

test["diff_tfidf_L2_norm"] = diff_tfidf_L2_norm_test

test["cos_tfidf"] = cos_tfidf_test



del tfidf_q1_test, tfidf_q2_test, diff, diff_tfidf_L1_test, diff_tfidf_L2_test

del diff_tfidf_L1_norm_test, diff_tfidf_L2_norm_test, cos_tfidf_test

del count1, count2, count3, count4
tr = pd.read_csv("../input/quora-question-pairs/train.csv")

te = pd.read_csv("../input/quora-question-pairs/test.csv")



ques = pd.concat([tr[["question1", "question2"]], te[["question1", "question2"]]], 

                 axis = 0).reset_index(drop = "index")

q_dict = defaultdict(set)

for i in range(ques.shape[0]):

        q_dict[ques.question1[i]].add(ques.question2[i])

        q_dict[ques.question2[i]].add(ques.question1[i])



def q1_q2_intersect(row):

    return len(set(q_dict[row['question1']]).intersection(set(q_dict[row['question2']])))



train["q1_q2_intersect"] = tr.apply(q1_q2_intersect, axis=1, raw=True)

test["q1_q2_intersect"] = te.apply(q1_q2_intersect, axis=1, raw=True)



del ques
def q1_freq(row):

    return(len(q_dict[row["question1"]]))

def q2_freq(row):

    return(len(q_dict[row["question2"]]))



train["q1_freq"] = tr.apply(q1_freq, axis=1, raw=True)

train["q2_freq"] = tr.apply(q2_freq, axis=1, raw=True)

train["q1_q2_freq_average"] = (train["q1_freq"] + train["q2_freq"]) / 2



test["q1_freq"] = te.apply(q1_freq, axis=1, raw=True)

test["q2_freq"] = te.apply(q2_freq, axis=1, raw=True)

test["q1_q2_freq_average"] = (test["q1_freq"] + test["q2_freq"]) / 2



del tr, te, q_dict
def same_start_word(row):

    wordlist1 = word_tokenize(row["question1"])

    wordlist2 = word_tokenize(row["question2"])

    if wordlist1 and wordlist2:  # 为了防止question1或question2是只包含分隔符的空问题

        return int(wordlist1[0] == wordlist2[0])

    else:

        return 0



train["same_start_word"] = train_orig.apply(same_start_word, axis = 1)

test["same_start_word"] = test_orig.apply(same_start_word, axis = 1)
sentiment_analyzer = SentimentIntensityAnalyzer()

def sentiment_analyze(row):

    sen1 = sentiment_analyzer.polarity_scores(row["question1"])

    sen2 = sentiment_analyzer.polarity_scores(row["question2"])

    diff_neg = np.abs(sen1["neg"] - sen2["neg"])

    diff_neu = np.abs(sen1["neu"] - sen2["neu"])

    diff_pos = np.abs(sen1["pos"] - sen2["pos"])

    diff_com = np.abs(sen1["compound"] - sen2["compound"])

    return pd.Series([diff_neg, diff_neu, diff_pos, diff_com])
sen_train = train_orig.apply(sentiment_analyze, axis = 1)

sen_train.columns = ["diff_sen_neg", "diff_sen_neu", "diff_sen_pos", "diff_sen_com"]

train = pd.concat([train, sen_train], axis = 1)



sen_test = test_orig.apply(sentiment_analyze, axis = 1)

sen_test.columns = ["diff_sen_neg", "diff_sen_neu", "diff_sen_pos", "diff_sen_com"]

test = pd.concat([test, sen_test], axis = 1)



del sen_train, sen_test
train.to_csv("train.csv", index = False)

test.to_csv("test.csv", index = False)

trainlabel.to_csv("trainlabel.csv", index = False)
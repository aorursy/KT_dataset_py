# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#read data

train_df = pd.read_csv('../input/train.csv')

test_df  = pd.read_csv('../input/test.csv')



#get an idea

train_df.info()

test_df.info()

train_df.head()

test_df.head()

#use csv file directly to create features for faster operatio

# refrences : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb

# https://www.kaggle.com/the1owl/matching-que-for-quora-end-to-end-0-33719-pb



import os, sys, re, csv

import pandas as pd

import numpy as np

from nltk.corpus import stopwords

from collections import Counter

from difflib import SequenceMatcher

import math

import nltk

from datetime import datetime



sw = set(stopwords.words('english'))



def get_cosine(str1, str2):



    vec1 = Counter(str1.split())

    vec2 = Counter(str2.split())



    intersection = set(vec1.keys()) & set(vec2.keys())

    numerator = sum([vec1[x] * vec2[x] for x in intersection])



    sum1 = sum([vec1[x]**2 for x in vec1.keys()])

    sum2 = sum([vec2[x]**2 for x in vec2.keys()])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)



    if not denominator:

        return 0.0

    else:

        return float(numerator) / denominator



def DistJaccard(str1, str2):

    str1 = set(str1.split())

    str2 = set(str2.split())

    numerator = len(str1 & str2)

    denominator = len(str1 | str2)

    if not denominator:

        return 0.0

    else:

        return float(numerator) / denominator



def similar(a, b):

    return SequenceMatcher(None, a, b).ratio()



def getAlphabetCount(word):

    alphabet_dict = {}

    txt = re.sub('[^A-Za-z]','',re.sub(' ','',word))

    for aa in txt:

        if aa in alphabet_dict:

            alphabet_dict[aa] += 1

        else:

            alphabet_dict[aa] = 1

    list1 = list(txt)

    list2 = list('abcdefghijklmnopqrstuvwxyz')

    diff = set(list2).symmetric_difference(list1)

    for bb in diff:

        alphabet_dict[bb] = 0

    return alphabet_dict



def decisionMaker(a,b):

    if(a==b):

        return 1

    else:

        return 0



def word_match_share(txt1, txt2):

    q1words = {}

    q2words = {}

    for word in txt1:

        q1words[word] = 1

    for word in txt2:

        q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    match_score = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))

    return match_score



def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1.0 / (count + eps)



eps = 5000



def getAlphabetCount(word):

    alphabet_dict = {}

    txt = re.sub('[^A-Za-z]','',re.sub(' ','',word))

    for aa in txt:

        if aa in alphabet_dict:

            alphabet_dict[aa] += 1

        else:

            alphabet_dict[aa] = 1

    list1 = list(txt)

    list2 = list('abcdefghijklmnopqrstuvwxyz')

    diff = set(list2).symmetric_difference(list1)

    for bb in diff:

        alphabet_dict[bb] = 0

    return alphabet_dict



def decisionMaker(a,b):

    if(a==b):

        return 1

    else:

        return 0



def word_match_share(txt1, txt2):

    q1words = {}

    q2words = {}

    for word in txt1:

        q1words[word] = 1

    for word in txt2:

        q2words[word] = 1

    if len(q1words) == 0 or len(q2words) == 0:

        return 0

    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]

    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]

    match_score = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))

    return match_score



def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1.0 / (count + eps)



eps = 5000



#This function is messed up - can be written much better

def tfidf_word_match_share(txt1, txt2):

    tfidf = []

    q1words = {}

    q2words = {}

    for word in txt1:

        q1words[word] = 1

    for word in txt2:

        q2words[word] = 1

    if len(q1words) == 0 or len(q1words) == 0 :

        return 0,0,0,0,0,0,0,0,0,0,0

    words = txt1 + txt2

    counts = Counter(words)

    weights = {word: get_weight(count) for word, count in counts.items()}

    q1_tfidf_weights = [weights.get(w, 0) for w in q1words.keys()]

    q2_tfidf_weights = [weights.get(w, 0) for w in q2words.keys()]

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]

    total_weights  = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    tfidfRatio = np.sum(shared_weights)*1.0 / np.sum(total_weights)

    q1_tfidf_sum     = sum(q1_tfidf_weights)

    q2_tfidf_sum     = sum(q2_tfidf_weights)

    q1_tfidf_mean    = np.mean(q1_tfidf_weights)

    q2_tfidf_mean    = np.mean(q2_tfidf_weights)

    q1_tfidf_min     = min(q1_tfidf_weights)

    q1_tfidf_max     = min(q1_tfidf_weights)

    q1_tfidf_range   = q1_tfidf_max - q1_tfidf_min

    q2_tfidf_min     = min(q2_tfidf_weights)

    q2_tfidf_max     = min(q2_tfidf_weights)

    q2_tfidf_range   = q2_tfidf_max - q2_tfidf_min

    return q1_tfidf_sum, q2_tfidf_sum, q1_tfidf_mean, q2_tfidf_mean, q1_tfidf_min, q1_tfidf_max, q1_tfidf_range, q2_tfidf_min, q2_tfidf_max, q2_tfidf_range, q2_tfidf_range

    

def shared_2gram(q1, q2):

    q1_2gram = set([i for i in zip(q1.split(), q1.split()[1:])])

    q2_2gram = set([i for i in zip(q2.split(), q2.split()[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    if len(q1_2gram)==0 or len(q2_2gram) == 0:

        return 0

    return len(shared_2gram)*1.0/(len(q1_2gram) + len(q2_2gram))



def shared_3gram(q1, q2):

    q1_3gram = set([i for i in zip(q1.split(), q1.split()[1:], q1.split()[2:])])

    q2_3gram = set([i for i in zip(q2.split(), q2.split()[1:], q2.split()[2:])])

    shared_3gram = q1_3gram.intersection(q2_3gram)

    if len(q1_3gram)==0 or len(q2_3gram) == 0:

        return 0

    return len(shared_3gram)*1.0/(len(q1_3gram) + len(q2_3gram))



def avgWordLenDiff(q1, q2):

    len_char_q1 = len(q1.replace(' ',''))

    len_char_q2 = len(q2.replace(' ',''))

    len_word_q1 = len(q1.split())

    len_word_q2 = len(q2.split())

    if len_char_q1 == 0 or len_char_q2 == 0:

        return 0, 0, 0

    avg_world_len_q1 = len_char_q1*1.0/len_word_q1

    avg_world_len_q2 = len_char_q2*1.0/len_word_q2

    return avg_world_len_q1, avg_world_len_q2, avg_world_len_q1-avg_world_len_q2



stops = set(stopwords.words("english"))

def avgStopWords(q1, q2):

    q1stops_len = len(set(q1.split()).intersection(stops))

    q2stops_len = len(set(q2.split()).intersection(stops))

    q1word_len  = len(q1.split())

    q2word_len  = len(q2.split())

    if q1stops_len == 0 or q2stops_len == 0:

        return 0,0,0

    q1_r = q1stops_len*1.0/q1word_len

    q2_r = q2stops_len*1.0/q2word_len

    q_diff = q1_r - q2_r

    return q1_r, q2_r, q_diff



def create_data(infile,outfile):

    

    if('train' in infile):

        with open('../input/' + infile) as file:

            reader = csv.reader(file, delimiter = ',')

            i = 0

            for line in reader:

                q1 = [re.sub('[^A-Za-z0-9]','',x).lower() for x in line[3].split() if len(x)>1 and x not in sw]

                q2 = [re.sub('[^A-Za-z0-9]','',x).lower() for x in line[4].split() if len(x)>1 and x not in sw]

                if len(q1) > 1 :

                    q1 = ' '.join([re.sub('[^A-Za-z0-9]','',x).lower() for x in q1 if len(x)>1 and x not in sw])

                else:

                    q1 = 'blank'

                if len(q2):

                    q2 = ' '.join([re.sub('[^A-Za-z0-9]','',x).lower() for x in q2 if len(x)>1 and x not in sw])

                else:

                    q2 = 'blank'

                q1_len_with_space, q2_len_with_space  = len(q1), len(q2)

                q1_len_without_space, q2_len_without_space  = len(re.sub(' ','',q1)), len(re.sub(' ','',q2))

                q1_no_of_words, q2_no_of_words = len(q1.split()), len(q2.split())

                q1_no_of_uniq_words, q2_no_of_uniq_words = len(set(q1.split())), len(set(q2.split()))



                len_with_space_ind = decisionMaker(q1_len_with_space, q2_len_with_space)

                len_without_space_ind = decisionMaker(q1_len_without_space, q2_len_without_space)

                no_of_words_ind = decisionMaker(q2_no_of_words, q2_no_of_words)

                no_of_uniq_words_ind = decisionMaker(q2_no_of_uniq_words, q2_no_of_uniq_words)



                diff_len_with_space = q1_len_with_space - q2_len_with_space

                diff_len_without_space = q1_len_without_space - q2_len_without_space

                diff_no_of_words = q1_no_of_words - q2_no_of_words

                diff_no_of_uniq_words = q1_no_of_uniq_words - q2_no_of_uniq_words



                alphabetCountDictQ1 = getAlphabetCount(q1)

                alphabetCountDictQ2 = getAlphabetCount(q2)

                q1_a, q2_a = alphabetCountDictQ1['a'], alphabetCountDictQ2['a']

                q1_b, q2_b = alphabetCountDictQ1['b'], alphabetCountDictQ2['b']

                q1_c, q2_c = alphabetCountDictQ1['c'], alphabetCountDictQ2['c']

                q1_d, q2_d = alphabetCountDictQ1['d'], alphabetCountDictQ2['d']

                q1_e, q2_e = alphabetCountDictQ1['e'], alphabetCountDictQ2['e']

                q1_f, q2_f = alphabetCountDictQ1['f'], alphabetCountDictQ2['f']

                q1_g, q2_g = alphabetCountDictQ1['g'], alphabetCountDictQ2['g']

                q1_h, q2_h = alphabetCountDictQ1['h'], alphabetCountDictQ2['h']

                q1_i, q2_i = alphabetCountDictQ1['i'], alphabetCountDictQ2['i']

                q1_j, q2_j = alphabetCountDictQ1['j'], alphabetCountDictQ2['j']

                q1_k, q2_k = alphabetCountDictQ1['k'], alphabetCountDictQ2['k']

                q1_l, q2_l = alphabetCountDictQ1['l'], alphabetCountDictQ2['l']

                q1_m, q2_m = alphabetCountDictQ1['m'], alphabetCountDictQ2['m']

                q1_n, q2_n = alphabetCountDictQ1['n'], alphabetCountDictQ2['n']

                q1_o, q2_o = alphabetCountDictQ1['o'], alphabetCountDictQ2['o']

                q1_p, q2_p = alphabetCountDictQ1['p'], alphabetCountDictQ2['p']

                q1_q, q2_q = alphabetCountDictQ1['q'], alphabetCountDictQ2['q']

                q1_r, q2_r = alphabetCountDictQ1['r'], alphabetCountDictQ2['r']

                q1_s, q2_s = alphabetCountDictQ1['s'], alphabetCountDictQ2['s']

                q1_t, q2_t = alphabetCountDictQ1['t'], alphabetCountDictQ2['t']

                q1_u, q2_u = alphabetCountDictQ1['u'], alphabetCountDictQ2['u']

                q1_v, q2_v = alphabetCountDictQ1['v'], alphabetCountDictQ2['v']

                q1_w, q2_w = alphabetCountDictQ1['w'], alphabetCountDictQ2['w']

                q1_x, q2_x = alphabetCountDictQ1['x'], alphabetCountDictQ2['x']

                q1_y, q2_y = alphabetCountDictQ1['y'], alphabetCountDictQ2['y']

                q1_z, q2_z = alphabetCountDictQ1['z'], alphabetCountDictQ2['z']



                a_count_ind = decisionMaker(q1_a, q2_a)

                b_count_ind = decisionMaker(q1_b, q2_b)

                c_count_ind = decisionMaker(q1_c, q2_c)

                d_count_ind = decisionMaker(q1_d, q2_d)

                e_count_ind = decisionMaker(q1_e, q2_e)

                f_count_ind = decisionMaker(q1_f, q2_f)

                g_count_ind = decisionMaker(q1_g, q2_g)

                h_count_ind = decisionMaker(q1_h, q2_h)

                i_count_ind = decisionMaker(q1_i, q2_i)

                j_count_ind = decisionMaker(q1_j, q2_j)

                k_count_ind = decisionMaker(q1_k, q2_k)

                l_count_ind = decisionMaker(q1_l, q2_l)

                m_count_ind = decisionMaker(q1_m, q2_m)

                n_count_ind = decisionMaker(q1_n, q2_n)

                o_count_ind = decisionMaker(q1_o, q2_o)

                p_count_ind = decisionMaker(q1_p, q2_p)

                q_count_ind = decisionMaker(q1_q, q2_q)

                r_count_ind = decisionMaker(q1_r, q2_r)

                s_count_ind = decisionMaker(q1_s, q2_s)

                t_count_ind = decisionMaker(q1_t, q2_t)

                u_count_ind = decisionMaker(q1_u, q2_u)

                v_count_ind = decisionMaker(q1_v, q2_v)

                w_count_ind = decisionMaker(q1_w, q2_w)

                x_count_ind = decisionMaker(q1_x, q2_x)

                y_count_ind = decisionMaker(q1_y, q2_y)

                z_count_ind = decisionMaker(q1_z, q2_z)

                

                diff_a_count = q1_a - q2_a

                diff_b_count = q1_b - q2_b

                diff_c_count = q1_c - q2_c

                diff_d_count = q1_d - q2_d

                diff_e_count = q1_e - q2_e

                diff_f_count = q1_f - q2_f

                diff_g_count = q1_g - q2_g

                diff_h_count = q1_h - q2_h

                diff_i_count = q1_i - q2_i

                diff_j_count = q1_j - q2_j

                diff_k_count = q1_k - q2_k

                diff_l_count = q1_l - q2_l

                diff_m_count = q1_m - q2_m

                diff_n_count = q1_n - q2_n

                diff_o_count = q1_o - q2_o

                diff_p_count = q1_p - q2_p

                diff_q_count = q1_q - q2_q

                diff_r_count = q1_r - q2_r

                diff_s_count = q1_s - q2_s

                diff_t_count = q1_t - q2_t

                diff_u_count = q1_u - q2_u

                diff_v_count = q1_v - q2_v

                diff_w_count = q1_w - q2_w

                diff_x_count = q1_x - q2_x

                diff_y_count = q1_y - q2_y

                diff_z_count = q1_z - q2_z



                cos_sim = get_cosine(q1, q2)

                jac_sim = DistJaccard(q1, q2)

                seq_mat = similar(q1, q2)

                #lav_dis = distance.levenshtein(q1, q2)

                word_match = word_match_share(q1, q2)

                tfidf_match = tfidf_word_match_share(q1, q2)[10]

                shared_2grams = shared_2gram(q1, q2)

                shared_3grams = shared_3gram(q1, q2)

                avg_world_len_q1 = avgWordLenDiff(q1, q2)[0]

                avg_world_len_q2 = avgWordLenDiff(q1, q2)[1]

                avg_world_diff   = avgWordLenDiff(q1, q2)[2]

                q1_stop_ratio    = avgStopWords(q1, q2)[0]

                q2_stop_ratio    = avgStopWords(q1, q2)[1]

                ratio_diff       = avgStopWords(q1, q2)[2]

                caps_count_q1 = sum([1 for j in line[3] if j.isupper()])

                caps_count_q2 = sum([1 for j in line[4] if j.isupper()])

                caps_count_diff = caps_count_q1 - caps_count_q2

                qmarks_q1 = sum([1 for j in line[3] if j=='?'])

                qmarks_q2 = sum([1 for j in line[4] if j=='?'])

                qmarks_diff = qmarks_q1 - qmarks_q2

                fs_q1 = sum([1 for j in line[3] if j=='.'])

                fs_q2 = sum([1 for j in line[4] if j=='.'])

                fs_diff = qmarks_q1 - qmarks_q2

                if(len(line[3])>1):

                    first_caps_count_q1 = sum([1 if line[3][0].isupper() else 0])

                else:

                    first_caps_count_q1 = 0

                if(len(line[4])>1):

                    first_caps_count_q2 = sum([1 if line[4][0].isupper() else 0])

                else:

                    first_caps_count_q2 = 0

                first_caps_count_diff = first_caps_count_q1 - first_caps_count_q2

                numb_count_q1 = sum([1 for j in line[3] if j.isdigit()])

                numb_count_q2 = sum([1 for j in line[4] if j.isdigit()])

                numb_count_diff = numb_count_q1 - numb_count_q2

                nouns_q1 = [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(q1).lower())) if t[:1] in ['N']]

                nouns_q2 = [w for w, t in nltk.pos_tag(nltk.word_tokenize(str(q2).lower())) if t[:1] in ['N']]

                noun_count_q1 = len(nouns_q1)

                noun_count_q2 = len(nouns_q2)

                noun_match    = sum([1 for w in nouns_q1 if w in nouns_q2])

                tfidf = tfidf_word_match_share(q1, q2)

                q1_tfidf_sum = tfidf[0]

                q2_tfidf_sum = tfidf_word_match_share(q1, q2)[1]

                q1_tfidf_mean = tfidf_word_match_share(q1, q2)[2]

                q2_tfidf_mean = tfidf_word_match_share(q1, q2)[3]

                q1_tfidf_min = tfidf_word_match_share(q1, q2)[4]

                q1_tfidf_max = tfidf_word_match_share(q1, q2)[5]

                q1_tfidf_range = tfidf_word_match_share(q1, q2)[6]

                q2_tfidf_min = tfidf_word_match_share(q1, q2)[7]

                q2_tfidf_max = tfidf_word_match_share(q1, q2)[8]

                q2_tfidf_range = tfidf_word_match_share(q1, q2)[9]



    #outfile.write(line[0] + '^' + q1 + '^' + q2 + '^' + str(len_with_space_ind) + '^' + str(len_without_space_ind) + '^' + str(no_of_words_ind) + '^' + str(no_of_uniq_words_ind) + '^' + str(diff_len_with_space) + '^' + str(diff_len_without_space) + '^' + str(diff_no_of_words) + '^' +  str(diff_no_of_uniq_words) + '^' + str(a_count_ind) + '^' + str(b_count_ind) + '^' + str(c_count_ind) + '^' + str(d_count_ind) + '^' + str(e_count_ind) + '^' + str(f_count_ind) + '^' + str(g_count_ind) + '^' + str(h_count_ind) + '^' + str(i_count_ind) + '^' + str(j_count_ind) + '^' + str(k_count_ind) + '^' + str(l_count_ind) + '^' + str(m_count_ind) + '^' + str(n_count_ind) + '^' + str(o_count_ind) + '^' + str(p_count_ind) + '^' + str(q_count_ind) + '^' + str(r_count_ind) + '^' + str(s_count_ind) + '^' + str(t_count_ind) + '^' + str(u_count_ind) + '^' + str(v_count_ind) + '^' + str(w_count_ind) + '^' + str(x_count_ind) + '^' + str(y_count_ind) + '^' + str(z_count_ind) + '^' + str(diff_a_count) + '^' + str(diff_b_count) + '^' + str(diff_c_count) + '^' + str(diff_d_count) + '^' + str(diff_e_count) + '^' + str(diff_f_count) + '^' + str(diff_g_count) + '^' + str(diff_h_count) + '^' + str(diff_i_count) + '^' + str(diff_j_count) + '^' + str(diff_k_count) + '^' + str(diff_l_count) + '^' + str(diff_m_count) + '^' + str(diff_n_count) + '^' + str(diff_o_count) + '^' + str(diff_p_count) + '^' + str(diff_q_count) + '^' + str(diff_r_count) + '^' + str(diff_s_count) + '^' + str(diff_t_count) + '^' + str(diff_u_count) + '^' + str(diff_v_count) + '^' + str(diff_w_count) + '^' + str(diff_x_count) + '^' + str(diff_y_count) + '^' + str(diff_z_count) + '^' + str(cos_sim) + '^' + str(jac_sim) + '^' + str(seq_mat) + '^' + str(lav_dis) + '^' + str(word_match) + '^' + str(tfidf_match) + '^' + str(shared_2grams) + '^' + str(shared_3grams) + '^' + str(avg_world_len_q1) + '^' + str(avg_world_len_q2) + '^' + str(avg_world_diff) + '^' + str(q1_stop_ratio) + '^' + str(q1_stop_ratio) + '^' + str(ratio_diff) + '^' + str(caps_count_q1) + '^' + str(caps_count_q2) + '^' + str(caps_count_diff) + '^' + str(qmarks_q1) + '^' + str(qmarks_q2) + '^' + str(qmarks_diff) + '^' + str(fs_q1) + '^' + str(fs_q2) + '^' + str(qmarks_diff) + '^' + str(first_caps_count_q1) + '^' + str(first_caps_count_q2) + '^' + str(first_caps_count_diff) + str(numb_count_q1) + '^' + str(numb_count_q2) + '^' + str(numb_count_diff) + '^' + str(noun_count_q1) + '^' + str(noun_count_q2) + '^' + str(noun_match) + '^' + str(q1_tfidf_sum) + '^' + str(q2_tfidf_sum) + '^' + str(q1_tfidf_mean) + '^' + str(q2_tfidf_mean) + '^' + str(q1_tfidf_min) + '^' + str(q1_tfidf_max) + '^' + str(q1_tfidf_range) + '^' + str(q2_tfidf_min) + '^' + str(q2_tfidf_max) + '^' + str(q2_tfidf_range) + '^' + str(line[5]) + '\n')              
create_data('train.csv', 'train_ftrs.csv')
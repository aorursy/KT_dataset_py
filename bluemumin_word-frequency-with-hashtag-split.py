import pandas as pd # 데이터 전처리

import numpy as np # 데이터 전처리

import random #데이터 전처리



from pandas import DataFrame #데이터 전처리

from collections import Counter #데이터 전처리



import re

import nltk

nltk.download('words')

from nltk.corpus import words, brown
word_dictionary = list(set(words.words()))

for alphabet in "bcdefghjklmnopqrstuvwxyz":

    word_dictionary.remove(alphabet)



def split_hashtag_to_words_all_possibilities(hashtag):

    all_possibilities = []



    split_posibility = [hashtag[:i] in word_dictionary for i in reversed(range(len(hashtag)+1))]

    possible_split_positions = [i for i, x in enumerate(split_posibility) if x == True]



    for split_pos in possible_split_positions:

        split_words = []

        word_1, word_2 = hashtag[:len(hashtag)-split_pos], hashtag[len(hashtag)-split_pos:]



        if word_2 in word_dictionary:

            split_words.append(word_1)

            split_words.append(word_2)

            all_possibilities.append(split_words)



            another_round = split_hashtag_to_words_all_possibilities(word_2)



            if len(another_round) > 0:

                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]

        else:

            another_round = split_hashtag_to_words_all_possibilities(word_2)



            if len(another_round) > 0:

                all_possibilities = all_possibilities + [[a1] + a2 for a1, a2, in zip([word_1]*len(another_round), another_round)]

                

    return all_possibilities
def print_3(original):

    word_space=[]

    for i in original:

        if len(i)<=3:

            word_space.append(i)

    return word_space
def print_er(original_word):

    word_space2=[]

    for j in original_word:

        if (len(j)==3) & ( len(j[-1])<=3 ):

            temp=[]

            temp.append( j[0] )

            temp.append( j[1]+j[2])

            word_space2.append(temp)

        else:

            word_space2.append(j)

    word_space=[]

    for i in word_space2:

        if 'er' in i:

            pass

        else:

            word_space.append(i)

    return [list(t) for t in set(tuple(element) for element in word_space)]



def print_ing(original_word):

    q=p = re.compile('ing$')

    word_space2=[]

    for j in original_word:

        if (len(j)==3) & ( len(j[-1])<=4 ):

            temp=[]

            temp.append( j[0] )

            temp.append( j[1]+j[2])

            word_space2.append(temp)

        elif (len(j)==3) & ( q.findall(j[-2])==['ing'] ):

            temp=[]

            temp.append( j[0]+j[1] )

            temp.append( j[2])

            word_space2.append(temp)

        else:

            word_space2.append(j)

    word_space=[]

    for i in word_space2:

        if 'ing' in i:

            pass

        else:

            word_space.append(i)

    return [list(t) for t in set(tuple(element) for element in word_space)]



def print_ed(original_word):

    word_space2=[]

    for j in original_word:

        if (len(j)==3) & ( len(j[-1])<=3 ):

            temp=[]

            temp.append( j[0] )

            temp.append( j[1]+j[2])

            word_space2.append(temp)

        else:

            word_space2.append(j)

    word_space=[]

    for i in word_space2:

        if 'ed' in i:

            pass

        else:

            word_space.append(i)

    return [list(t) for t in set(tuple(element) for element in word_space)]
def print_man(original_word):

    raw=''.join(original_word[0])

    original_word.append( [ raw[:-3],raw[-3:] ]  )



    return [list(t) for t in set(tuple(element) for element in original_word)]



def print_wm(original_word):

    raw=''.join(original_word[0])

    original_word.append( [ raw[:-5],raw[-5:] ]  )



    return [list(t) for t in set(tuple(element) for element in original_word)]
count_v = pd.read_csv('../input/english-word-frequency/unigram_freq.csv')

count_v['type'] = [type(i) for i in count_v['word']]

count_v2=count_v[count_v['type']==str]

count_v2['len'] = [len(i) for i in count_v2['word']]



count_v2=count_v2[count_v2['len']>=2]
count_v2.head(10)
remove_list=['to','in','by','go','of','in','on','as','the','and','up']

def regulatoin_list(next_word):

    word_space=[]

    if next_word==[]:

        return [next_word]

    else:

        for list1 in next_word:

            word_space2 = [len(i) for i in list1]

            if (1 in word_space2) :  #길이가 하나라도 1인 경우

                pass

            elif (word_space2.count(2)==2) :  # 2개 단어의 길이가 2인 경우는 비정상적이므로 제외

                pass

            else:

                word_space.append(list1)



        if len(word_space)>=2:

            sum_list=[]

            real_list=[]

            for splitting in word_space:

                if len(splitting)==2:

                    if ( (len(splitting[-1])==2) & ( splitting[-1] not in remove_list ) ) | ( ( len(splitting[-2])==2 ) & ( splitting[-2] not in remove_list ) ) :

                        pass

                    else:

                        real_list.append(splitting)

                else:

                    if ( len(splitting[-1])==2 ) | ( ( len(splitting[-2])==2 ) & ( splitting[-2] not in remove_list ) ) | ( ( len(splitting[-3])==2 ) & ( splitting[-3] not in remove_list ) ) :

                        pass

                    else:

                        real_list.append(splitting)



            for j in real_list:

                sum1 = 1

                for y in range(len(j)):

                    try:

                        sum1 += count_v[count_v['word']==j[y]].index[0]

                    except:

                        sum1 += 99999999

                sum_list.append(sum1)

            return real_list[ sum_list.index(min(sum_list)) ]

            

        elif len(word_space)==0:

            return []



        else:

            return word_space[0]
def word_space(j):

    p = re.compile('er$')

    

    if p.findall(j)==['er']:

        try:

            return regulatoin_list( print_er( print_3( split_hashtag_to_words_all_possibilities(j) ) ) ) 

        except:

            return [j]



    elif j.find("ing")>(-1):

        try:

            return regulatoin_list( print_ing( print_3( split_hashtag_to_words_all_possibilities(j) ) ) )

        except:

            return [j]



    elif j[-5:]=="woman":

        try:

            return regulatoin_list( print_wm( print_3( split_hashtag_to_words_all_possibilities(j) ) ) )

        except:

            return [j]



    elif (j[-3:]=="man") &  (j[-5:]!="woman"):

        try:

            return regulatoin_list( print_man( print_3( split_hashtag_to_words_all_possibilities(j) ) ) )

        except:

            return [j]

    elif j[-2:]=="ed" :

        try:

            return regulatoin_list( print_ed( print_3( split_hashtag_to_words_all_possibilities(j) ) ) )

        except:

            return [j]

    

    else:

        try:

            return regulatoin_list( print_3( split_hashtag_to_words_all_possibilities( j ) ) )

        except:

            return [j]
print( word_space('snowman') )



print( word_space('longwinded') )



print( word_space('hashtagsplit') )



print( word_space('strawberry') )



print( word_space('strawberrycake') )



print( word_space('blueberrycake') )



print( word_space('watermelonsugar'))



print( word_space('watermelonsugarsalt'))



print( word_space('themselves'))
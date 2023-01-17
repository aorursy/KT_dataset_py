import collections

from collections import defaultdict #импортируем словарь для спам и неспам слов

import re #для очистки текстов

import pandas as pd #для работы с обучающей выборкой

import numpy as np

import math
SPAM = 1

NOT_SPAM = 0



pA = 0.0 #вероятность встретить спам

pNotA = 0.0 #вероятность не встретить спам

YesSpam = {} #словарь для спамслов

NoSpam = {} #словарь для неспам-слов



allword={} # словарь для всех слов в обоих выборках



class_freqs = collections.defaultdict(int) #словарь для количества классов

class_freqs[SPAM]=0

class_freqs[NOT_SPAM]=0



url_data = '../input/spam-or-not-spam-dataset/spam_or_not_spam.csv'
def get_dict(label):

    if (label == 1):

        return YesSpam

    else:

        return NoSpam
remove_non_alphabets =lambda x: re.sub(r'[^a-zA-Z]',' ',x)
def calculate_word_frequencies(body, label):

    

    text = body.lower()



    list_of_str = re.findall(r'\b[a-z]+\b', text) #зададим ограничение на длину слова, чтобы исключить предлоги и тп

    

    dict1 = get_dict(label)

    

    for word in list_of_str:

        if (word in dict1):

            dict1[word] = dict1[word] + 1

        else:

            dict1[word] = 1

    

    for word in list_of_str:

        if word in allword.keys():

            allword[word] += 1

        else:

            allword[word] = 1

    return
def train(train_data):

    global YesSpam

    global NotSpam 

    global pA 

    global pNotA 

    

    YesSpam = {}

    NotSpam = {}

    

    for data in train_data:

        calculate_word_frequencies(data[0], data[1])

        class_freqs[data[1]] += 1

    

    pA = class_freqs[SPAM]/(class_freqs[SPAM] + class_freqs[NOT_SPAM])

    pNotA = class_freqs[NOT_SPAM]/(class_freqs[SPAM] + class_freqs[NOT_SPAM])



    return
def init_train():



    df = pd.read_csv(url_data)

# 0 - 2500 not spam 

# 1 - 500 is spam

    df.dropna(inplace=True)

    df['email'] = df['email'].apply(remove_non_alphabets)

    

    train_data = []

    

    df1 = df[df.label == 1]

    df0 = df[df.label == 0]



    for index, row in df1.iterrows():

        train_data.append([row['email'], SPAM])



    for index, row in df0.iterrows():

        train_data.append([row['email'], NOT_SPAM])

    

    train(train_data)

#train(train_data)
init_train()
#Код для отображения всех результатов ячеек в одном аутпуте для отладки

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
YesSpam
NoSpam
class_freqs
print('Spam words qty:', len(YesSpam),'NotSpam words qty:', len(NoSpam),'\n', sep='\n')

print('pA:', pA, 'pNotA:', pNotA, sep='\n')
def calculate_P_Bi_A(word, label): 

    dict1 = get_dict(label)

# Вероятность того, что слово является спамом:

# в числителе число вхождений слова в спам словарь

# в знаменателе число слов в словарях спам и не спам



    pt = 0

    if (word in dict1):

        pt = np.log((1+dict1[word])/(len(dict1)+sum(dict1.values())))

    else:

        pt = np.log(1/sum(allword.values()))

    return pt
def calculate_P_B_A(text, label):

    text_low = text.lower()

    words = re.findall(r'\b[a-z]+\b',text_low)

    

    result = 0

    for word in words:

        result += calculate_P_Bi_A(word, label)

    return result
def classify(email):

  

    p_A_B = np.log(pA) + calculate_P_B_A(email, SPAM)

    p_A_not_B = np.log(pNotA) + calculate_P_B_A(email, NOT_SPAM)

    

    print('prob_spam', p_A_B)

    print('prob_not_spam', p_A_not_B)

   

    if p_A_B > p_A_not_B:

        return 'SPAM'

    else:

        return 'NOT SPAM'
email = 'Hi, My name is Warren E. Buffett an American business magnate, investor and philanthropist. am the most successful investor in the world. I believe strongly in‘giving while living’ I had one idea that never changed in my mind? that you should use your wealth to help people and i have decided to give {$1,500,000.00} One Million Five Hundred Thousand United Dollars, to randomly selected individuals worldwide. On receipt of this email, you should count yourself as the lucky individual. Your email address was chosen online while searching at random. Kindly get back to me at your earliest convenience before i travel to japan for my treatment , so I know your email address is valid. Thank you for accepting our offer, we are indeed grateful You Can Google my name for more information: God bless you. Best Regard Mr.Warren E. Buffett Billionaire investor !'

classify(email)
email = "As a result of your application for the position of Data Engineer, I would like to invite you to attend an interview on May 30, at 9 a.m. at our office in Washington, DC. You will have an interview with the department manager, Moris Peterson. The interview will last about 45 minutes. If the date or time of the interview is inconvenient, please contact me by phone or email to arrange another appointment. We look forward to seeing you."

classify(email)
# SPAM

email = " attention this is a must for all computer users new special package deal norton systemworks NUMBER software suite professional edition includes six yes NUMBER feature packed utilities all for NUMBER special low price NUMBER feature packed utilities NUMBER great price a NUMBER combined retail value free shipping hyperlink click here now"

classify(email)
#NOT_SPAM

email = "i need to setup a vpn between a few sites from what i ve read the the choices come down on the linux side to ipsec using freeswan or cipe it seems that freeswan is better being an implementation of ipsec which is a standard however cipe does the job as well for linux clients and is somewhat simpler to setup the problem is that it s not a pure linux situation a couple of the sites run os x i m pretty sure that i ll be able to find an implementation of ipsec for os x but i think cipe is linux only so the question is for those of you have have implemented both is there a significant difference in setup time and hassle between cipe and freeswan if cipe is going to be much easier than dealing with freeswan and whatever on the os x sites then i ll simply get a linux box for each of the remote sites with the low price of hardware it doesn t take much more complexity in software to make buying hardware to use simpler software economic niall irish linux users group ilug URL URL for un subscription information list maintainer listmaster URL"

classify(email)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pA = .0 

pNotA = .0

trainP, trainN = {}, {}

totalP, totalN = 0, 0



url = '../input/spam-or-not-spam-dataset/spam_or_not_spam.csv'
remove_punctuation = lambda x: re.sub(r'[^A-Za-z]', ' ', x)

remove_noise = lambda x: re.sub(r'\b\w\b', '', x)

stopwords = ['number', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',

            'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',

            'she','her', 'hers', 'herself', 'its', 'itself', 'they', 'them', 'their', 

            'theirs', 'themselves', 'this', 'am', 'is', 'are', 'was', 'were', 'be', 

            'been', 'being', 'have', 'has', 'it', 'had', 'having', 'do', 'does', 'did', 

            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'as', 'until', 'of', 

            'at', 'by', 'for', 'with', 'to', 'on', 'is', 'in', 'al', 'at', 'then']
def calculate_word_frequencies(body, label):

    global trainP, trainN, totalP, totalN, stopwords

    

    # разбиваем предложение и удаляем слова, не несущие смысловой нагрузки

    for word in body.lower().split():

        if word not in stopwords:

            if label == 1:

                trainP[word] = trainP.get(word, 0) + 1

                totalP += 1

            else:

                trainN[word] = trainN.get(word, 0) + 1

                totalN += 1
def start_train():

    global df

    df = pd.read_csv(url)

    # Удаляем пустые строки, удаляем знаки препинания и слова длиной в 1 букву

    df.dropna(inplace = True)

    df['email'] = df['email'].apply(remove_punctuation)

    df['email'] = df['email'].apply(remove_noise)

    

    train_df = []



    for index, row in df.iterrows():

        if row['label'] == 1:

            train_df.append([row['email'], 1])

        else:

            train_df.append([row['email'], 0])

        

    train(train_df)





def train(data):

    global pNotA, pA

    spam_num = 0

    for (email, label) in data:

        calculate_word_frequencies(email, label)

        if label == 1:

            spam_num += 1

    pA = spam_num / len(data)

    pNotA = 1 - pA
# Рассчитываем вероятность спама для каждого слова со сглаживанием, 

# учитывая что в тестовой выборке встретим незнакомые для алгоритма слова

def calculate_P_Bi_A(word, label):

#     P(Bi|A) - вероятность для слова из текста

    vocabulary_size = len(set(trainP.keys()) | set(trainN.keys()))

    if label == 1:

        return np.log((trainP.get(word, 0) + 1) / (totalP + 1*vocabulary_size))

    else:

        return np.log((trainN.get(word, 0) + 1) / (totalN + 1*vocabulary_size))





def calculate_P_B_A(text, label):

#     P(B|A) - вероятность всего текста

    result = .0    

    for word in text.lower().split():

        if word not in stopwords:

            result += calculate_P_Bi_A(word, label)

    return result





def classify(email):

    isSpam = np.log(pA) + calculate_P_B_A(email, 1)

    notSpam = np.log(pNotA) + calculate_P_B_A(email, 0)

    

    print('Spam prob', isSpam)

    print('Not Spam prob', notSpam)

    if isSpam >= notSpam:

        return 'SPAM'

    else:

        return 'NOT SPAM'
start_train()

df.head()
print('Spam words:', len(trainP), 'Not Spam words:', len(trainN), '\n', sep = '\n')

print('Spam probability:', pA, 'Not Spam probability:', pNotA, sep='\n')
text_1 = "Hi, My name is Warren E. Buffett an American business magnate, investor and philanthropist. am the most successful investor in the world. I believe strongly in‘giving while living’ I had one idea that never changed in my mind? that you should use your wealth to help people and i have decided to give {$1,500,000.00} One Million Five Hundred Thousand United Dollars, to randomly selected individuals worldwide. On receipt of this email, you should count yourself as the lucky individual. Your email address was chosen online while searching at random. Kindly get back to me at your earliest convenience before i travel to japan for my treatment , so I know your email address is valid. Thank you for accepting our offer, we are indeed grateful You Can Google my name for more information: God bless you. Best Regard Mr.Warren E. Buffett Billionaire investor !"



text_2 = "I need to setup a vpn between a few sites from what i ve read the the choices come down on the linux side to ipsec using freeswan or cipe it seems that freeswan is better being an implementation of ipsec which is a standard however cipe does the job as well for linux clients and is somewhat simpler to setup the problem is that it s not a pure linux situation a couple of the sites run os x i m pretty sure that i ll be able to find an implementation of ipsec for os x but i think cipe is linux only so the question is for those of you have have implemented both is there a significant difference in setup time and hassle between cipe and freeswan if cipe is going to be much easier than dealing with freeswan and whatever on the os x sites then i ll simply get a linux box for each of the remote sites with the low price of hardware it doesn t take much more complexity in software to make buying hardware to use simpler software economic niall irish linux users group ilug URL URL for un subscription information list maintainer listmaster URL"



text_3 = "Attention this is a must for all computer users new special package deal norton systemworks NUMBER software suite professional edition includes six yes NUMBER feature packed utilities all for NUMBER special low price NUMBER feature packed utilities NUMBER great price a NUMBER combined retail value free shipping hyperlink click here now"
# SPAM

classify(text_1)
# NOT_SPAM

classify(text_2)
# SPAM

classify(text_3)
import re

import math

import pandas as pd
data_url = '../input/spam-or-not-spam-dataset/spam_or_not_spam.csv'



pA = 0  # probability of encountering SPAM

pNotA = 0  # probability of encountering NOT SPAM



## Consts

SPAM = 1

NOT_SPAM = 0

WORD_MIN_LENGTH = 3  # minimum separate word length to consider it valuable for model training



## Vars

trainPositive, trainNegative = {}, {}  # dictionaries for storing quantities of spam / non-spam words.

unique_words = 0  # number of unique words in general

totals = [0, 0]  # total amounts of non-spam [0] / spam [1] words



## Helper func to easily retrieve required dictionary by label

get_dict = lambda label: trainPositive if label == SPAM else trainNegative
def train(train_df):

    spam_count = 0

    total = len(train_df)

    for i in range(total):

        row = train_df.iloc[i]

        calculate_word_frequencies(row.email, row.label)

        spam_count += row.label

    global pA, pNotA, unique_words

    unique_words = len({*trainPositive, *trainNegative})

    pA = spam_count / total

    pNotA = 1 - pA
def calculate_word_frequencies(body, label):

    wordsDict = get_dict(label)

    for word in make_training_sample(body):

        wordsDict[word] = wordsDict.get(word, 0) + 1

        totals[label] += 1
def make_training_sample(text):

    s = re.sub('NUMBER', ' NUMBER ', text)

    words = []

    for w in re.findall(r'[a-zA-Zа-яА-ЯёЁ]+', s):  # get only words consisting of letters

        if len(w) >= WORD_MIN_LENGTH:  # satisfy the minimum valuable length of a word

            if w not in ['NUMBER', 'URL']:  # words NUMBER and URL are considered special masks

                w = w.lower()

            words.append(w)

    return words



def generalize_email(text):

    t = text.lower()

    t = re.sub(r'\d+[,.]{1}', lambda m: m.group()[:-1], t)  # bring digital numbers to general form

    t = re.sub(r'\d+', ' NUMBER ', t)  # replace all digital numbers with word 'NUMBER'

    t = re.sub(r'[^a-zA-Zа-яА-ЯёЁ]+', ' ', t)  # replace all non-letter characters with spaces

    return t
# P(Bi|A) - probability of finding word among SPAM (A) / NOT SPAM (^A)

def calculate_P_Bi_A(word, label):

    return (get_dict(label).get(word, 0) + 1) / (unique_words + totals[label])

    

# P(B|A) - probability of encountering text among SPAM (A) / NOT SPAM (^A)

def calculate_P_B_A(body, label):

    return sum([math.log(calculate_P_Bi_A(word, label)) for word in body.split()])
def classify(email):

    if 0 in totals:

        return 'ERROR: Not enough train data or model training failed!'

    email = generalize_email(email)

    isSpam = math.log(pA) + calculate_P_B_A(email, SPAM)

    isNotSpam = math.log(pNotA) + calculate_P_B_A(email, NOT_SPAM)

    spam_prob = 1 / (1 + math.exp(isNotSpam - isSpam))

    return ('SPAM' if isSpam > isNotSpam else 'NOT SPAM') + f' ::: Spam Probability: {str(spam_prob * 100)[0:6]} %'
df = pd.read_csv(data_url)

df = df.dropna()



train(df)
example1 = '''

Hi, My name is Warren E. Buffett an American business magnate, investor and philanthropist. 

am the most successful investor in the world. I believe strongly in‘giving while living’ 

I had one idea that never changed in my mind? that you should use your wealth to help people 

and i have decided to give {$1,500,000.00} One Million Five Hundred Thousand United Dollars, 

to randomly selected individuals worldwide. On receipt of this email, you should count yourself 

as the lucky individual. Your email address was chosen online while searching at random. 

Kindly get back to me at your earliest convenience before i travel to japan for my treatment , 

so I know your email address is valid. Thank you for accepting our offer, we are indeed grateful 

You Can Google my name for more information: God bless you. Best Regard Mr.Warren E. Buffett 

Billionaire investor !

'''



example2 = '''

Hi guys I want to build a website like REDACTED and I wanted to get your perspective of 

whether that site is good from the users' perspective before I go ahead and build something 

similar. I think that the design of the site is very modern and nice but I am not sure how 

people would react to a similar site? I look forward to your feedback. Many thanks!

'''



example3 = '''

As a result of your application for the position of Data Engineer, I would like to invite 

you to attend an interview on May 30, at 9 a.m. at our office in Washington, DC. You will 

have an interview with the department manager, Moris Peterson. The interview will last 

about 45 minutes. If the date or time of the interview is inconvenient, please contact me 

by phone or email to arrange another appointment. We look forward to seeing you.

'''



print('example1:', classify(example1))

print('example2:', classify(example2))

print('example3:', classify(example3))



print('\n'f'pA: {pA}')

print(f'total words: {totals}')

print(f'unique words: {unique_words}')

print('unique NOT_SPAM Dict words:', len(trainNegative), '\nunique SPAM Dict words:', len(trainPositive))
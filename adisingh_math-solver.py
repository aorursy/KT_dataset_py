import numpy as np 

import pandas as pd

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/mathQuestions.csv")



def removeStopWords(string):

    stop_words = set(stopwords.words('english'))

    word_tokens = word_tokenize(string)

    filtered_sentence = [w for w in word_tokens if not w in stop_words and len(w)>1]

    return filtered_sentence

'''

def n2w(n):

    num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', \

                 6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten', \

                11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', \

                15: 'Fifteen', 16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', \

                19: 'Nineteen', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', \

                50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty', \

                90: 'Ninety', 0: 'Zero'}





    try:

        print(num2words[n])

    except KeyError:

        try:

            print(num2words[n-n%10] + num2words[n%10].lower())

        except KeyError:

            print('Number out of range')

'''

for i in data["question"]:

    words = removeStopWords(i)

    tags = nltk.pos_tag(words)

    print(tags)

    

    
!pip install word2number

from word2number import w2n

numbers = [ 'zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve',

            'thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty',

            'thirty','forty','fifty','sixty','seventy','eighty','ninety','hundred','thousand']



for q in data["question"]:

    words = removeStopWords(i)

    for w in words:

        if w in numbers:

            n = w2n.word_to_num(w)

            print(n)
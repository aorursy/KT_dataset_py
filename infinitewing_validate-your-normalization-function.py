import pandas as pd

import numpy as np 

import re



df_train = pd.read_csv('../input/en_train.csv')

classes =  (pd.factorize(df_train['class'])[1]).values
def valid_norm(c='PLAIN', method=None):

    if(c not in classes):

        print('Unkown class')

        return

    if(method == None):

        print('Must pass transform method')

        return

    total_change = 0

    total_word = 0

    total_correct = 0

    df = df_train.loc[df_train['class'] == c]

    try:

        for i, row in df.iterrows():

            b = row['before']

            a = row['after']

            norm = method(b)

            total_word += 1

            if(b != norm):

                total_change += 1

                if(norm == a):

                    total_correct += 1

            else:

                pass

        try:

            print('Validation on {}'.format(c.lower()))

            print('{} / {}, changed rate = {}%'.format(total_change, total_word,total_change/total_word))    

            print('{} / {}, changed acc = {}%'.format(total_correct, total_change,total_correct/total_change))

            print('{} / {}, total acc = {}%'.format(total_correct, total_word,total_correct/total_word))

            

        except:

            print('exception: divided by 0')

            return

    except:

        print('transform method must be xxxx(s), which will return a string')

        return
def letters(x):

    try:

        x = re.sub('[^a-zA-Z]', '', x)

        x = x.lower()

        result_string = ''

        for i in range(len(x)):

            result_string = result_string + x[i] + ' '

        return(result_string.strip())  

    except:

        return x
valid_norm(c='LETTERS', method=letters)
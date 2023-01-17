#PLAIN, PUNCT data doesn't required to change

#some of DATE and all of CARDINAL are treated as numbers

#VERBATIM, PLAIN are treated the same because they don't require transformation

__author__ = 'BingQing Wei'

import re

import pandas as pd



NUMBER_TMP = r'^\d+$'

DECIMAL_TMP = r'^(?!0)\d*\.\d+$' #not start with 0

LETTER_TMP = r'^[A-Z]+\.*$'

DATE_TMP = r'^\d*\s*[a-zA-Z]+\s*\d*[\s,]+\d+$'

MEASURE_TMP = r'^\d*\.*\d*\s*[km%]+$'

MONEY_TMP = r'^\$.+$'

ORDINAL_TMP = r'^(\d+[rdsthn]+|[VIX]+\.*)$'

TIME_TMP = r'^(0\d+\.\d+|\d+:\d+|\d+)[\spma\.]*$'

ELECTRONIC_TMP = r'^(::|.*(\.[a-zA-Z]+|/.*))$'

FRACTION_TMP = r'^\d+/\d+$'

TELEPHONE_TMP = r'^[\d\s-]+$'

PLAIN_TMP = r'.*'
'''

patterns are matched in specific order according to cascade_order

'''



cascade_order = [NUMBER_TMP, DECIMAL_TMP,

                 DATE_TMP, MEASURE_TMP, MONEY_TMP,

                 ORDINAL_TMP, TIME_TMP, FRACTION_TMP,

                 ELECTRONIC_TMP, TELEPHONE_TMP,

                 LETTER_TMP, PLAIN_TMP]

tmp_dict = {NUMBER_TMP:'NUMBER', DECIMAL_TMP:'DECIMAL',

            LETTER_TMP:'LETTER', DATE_TMP:'DATE',

            MEASURE_TMP:'MEASURE', MONEY_TMP:'MONEY',

            ORDINAL_TMP:'ORDINAL', TIME_TMP:'TIME',

            ELECTRONIC_TMP:'ELECTRONIC', FRACTION_TMP:'FRACTION',

            TELEPHONE_TMP:'TELEPHONE',

            PLAIN_TMP:'PLAIN'}
'''

exampls to test regex

'''

test_examples = {'02.26':'TIME',

                 '3:00 am':'TIME',

                 '8 a.m.':'TIME',

                 '21st':'ORDINAL',

                 'V.':'ORDINAL',

                 '$29,583':'MONEY',

                 '60 km':'MEASURE',

                 '16.4%':'MEASURE',

                 '.161':'DECIMAL',

                 'IUCN.':'LETTER',

                 '4 March 2014':'DATE',

                 '978-0-646-34220-7':'TELEPHONE',

                 '192 1067-8':'TELEPHONE',

                 '12639/12640':'FRACTION',

                 'http://www.hkdailynews.com.hk/NewsDetail/index...':'ELECTRONIC',

                 'Rosettacode.org':'ELECTRONIC',

                 '::':'ELECTRONIC',

                 'November 4, 2014':'DATE',

                 'Brillantaisia':'PLAIN'}
'''

in the dataframe, 'Data' corresponds to the test examples

'Predict' is what our naive regex predicts

'Target' is the original label of the test examples

'''

def match():

    df = pd.DataFrame(columns=['Data', 'Predict', 'Target'])

    i = 0

    for exp in test_examples.keys():

        for pat in cascade_order:

            if re.match(pattern=pat, string=exp):

                df.loc[i] = [exp, tmp_dict[pat], test_examples[exp]]

                i = i + 1

                break

    print(df)



if __name__ == '__main__':

    match()
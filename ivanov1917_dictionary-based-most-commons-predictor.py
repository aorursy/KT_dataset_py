import numpy as np

import pandas as pd
train = pd.read_csv('../input/ru_train.csv')

test = pd.read_csv('../input/ru_test.csv')

test['id'] = test['sentence_id'].astype(str) + '_' + test['token_id'].astype(str)

# remove these 2 columns from the result table

test.drop(['sentence_id', 'token_id'], axis=1, inplace=True)
from collections import Counter



universe_dict_counter = dict() # counters

universe_dict = dict() # result dict

skipped_specials = '.-–—«»' # do not convert these characters



# populate universe_dict_counter with all possible before->after convertions

def populate_universe_dict_counter(row):

    global universe_dict_counter, skipped_specials

    

    before = str(row['before'])

    after = str(row['after'])

    if (before not in skipped_specials) and before != after:

        # create a Counter() for every word we found in a traing set

        if before not in universe_dict_counter:

            universe_dict_counter[before] = Counter()

        universe_dict_counter[before][after] += 1

        return False # return value doesn't matters



# get all converted rows

t = train[:10000] # limit to 10000 rows for this example

conversions = t[t['after'] != t['before']]

# ... and create a universe_dict_counter from it (axis=1 for rows):

conversions.apply(populate_universe_dict_counter, axis=1)



# get the most common before->after conversions

for word in universe_dict_counter:

    universe_dict[word] = universe_dict_counter[word].most_common(1)[0][0]
def convert(row):

    global universe_dict

    before = str(row['before'])

    if before in universe_dict:

        return universe_dict[before]

    else:

        return before



t = test[:10000] # limit to 1000 rows for this example

t['afters'] = t.apply(convert, axis=1)

del t['before'] # remove before column — it must no be in the result output
t.head(50)



# note the t[17]. It is converted from '2010 года' to 'две тысячи десятого года'
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import json



from IPython.display import display



#local script

from tfutils_py import get_answer, read_sample



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def preprocess(n=10):

    df = read_sample(n=n,ignore_doc_text=False)

    df['yes_no'] = df.annotations.apply(lambda x: x[0]['yes_no_answer'])

    df['long'] = df.annotations.apply(lambda x: [x[0]['long_answer']['start_token'], x[0]['long_answer']['end_token']])

    df['short'] = df.annotations.apply(lambda x: x[0]['short_answers'])

    return df

df = preprocess(5000)
# let's keep a mask of the short answers that exist.

mask_short_answer_exists = df.short.apply(lambda x: "Answer Doesn't exist" if x == [] else "Answer Exists") == "Answer Exists"
def is_short_in_long(text: str, short: dict, long: dict) -> bool:

    """

    Checks if a short answer is contained inside the long answer.

    """

    if short['start_token'] >= long['start_token'] and short['end_token'] <= long['end_token']:

        return True

    return False



def are_shorts_in_long(text: str, shorts: list, long: dict) -> list:

    """

    Checks for each short answer if they are contained in the long answer.

    

    ------

    Returns a list with the same size of the number of short answers and

    each position is a boolean determining if each short answer was contained 

    withing the long answer.

    """

    answer = []

    for short in shorts:

        answer.append(is_short_in_long(text,short,long))

    return answer



def are_all_shorts_in_long(text: str, shorts: list, long: dict) -> bool:

    """

    Gets a list of short answers and returns true if all of them were

    contained withing the long answer and false if otherwise.

    """

    shorts_in_long = are_shorts_in_long(text,shorts,long)

    if all(shorts_in_long):

        return True

    return False



short_in_long = df.loc[mask_short_answer_exists].apply(\

                    lambda row: are_all_shorts_in_long(row.document_text, \

                                                       row.annotations[0]['short_answers'], \

                                                       row.annotations[0]['long_answer']\

                                                      ),axis=1)

print("Are all of the short answers annotations always contained within the long_answer annotations?\n",\

      ("-> Yes" if all(short_in_long.values) else "-> No"))
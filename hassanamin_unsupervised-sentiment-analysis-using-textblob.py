# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from textblob import TextBlob

# Get the polarity score using below function

def get_textBlob_score(sent):

    # This polarity score is between -1 to 1

    polarity = TextBlob(sent).sentiment.polarity

    return polarity
get_textBlob_score("The phone is super cool.")
print("! ",get_textBlob_score("The phone is super cool!"))

print("!! ",get_textBlob_score("The phone is super cool!!"))

print("!!! ",get_textBlob_score("The phone is super cool!!!"))
print(get_textBlob_score("The phone is super cool."))

print(get_textBlob_score("The phone is super COOL."))
print("Food here is good. ",get_textBlob_score("Food here is good."))

print("Food here is moderately good. ",get_textBlob_score("Food here is moderately good."))

print("Food here is extremely good. ",get_textBlob_score("Food here is extremely good."))
get_textBlob_score("Food here is extremely good but service is horrible.")
get_textBlob_score("The food here isn’t really all that great")
print(get_textBlob_score('I am 😄 today'))

print(get_textBlob_score('😊'))

print(get_textBlob_score('😥'))

print(get_textBlob_score('☹️'))
print(get_textBlob_score("Today SUX!"))

print(get_textBlob_score("Today only kinda sux! But I'll get by, lol"))
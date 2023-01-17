# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv").dropna()

test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

train
def count(lis, text):

    count = 0

    for i in lis:

        if i.find(text) != -1:

            count += 1

    return count
count(train.text, '&quot;'), count(train.text, '&amp;'), count(train.text, '&lt;'), count(train.text, '&gt;')
count(train.selected_text, '&quot;'), count(train.selected_text, '&amp;'), count(train.selected_text, '&lt;'), count(train.selected_text, '&gt;')
count(test.text, '&quot;'), count(test.text, '&amp;'), count(test.text, '&lt;'), count(test.text, '&gt;')
count = 0

for i in train.text:

    i = i.replace("&quot;", '"')

    i = i.replace("&lt;", '<')

    i = i.replace("&gt;", '>')

    i = i.replace("&amp;", '')

    if i.find('&') != -1:

        count += 1

count
def remove_html_char_ref(i):

    i = i.replace("&quot;", '"')

    i = i.replace("&lt;", '<')

    i = i.replace("&gt;", '>')

    i = i.replace("&amp;", '&')

    return i
count = 0

for i,j, s in zip(train.text, train.selected_text, train.sentiment):

    if len(i) != len(remove_html_char_ref(i)): 

        count +=1

count
for i,j, s in zip(train.text, train.selected_text, train.sentiment):

    if len(i) != len(remove_html_char_ref(i)): 

        print("sentiment : "+s)

        print("original  : "+i+'[END]')

        print("corrected : "+remove_html_char_ref(i)+'[END]')

        print("selected  : "+j+'[END]')

        print("guess     : "+remove_html_char_ref(i)[i.find(j):i.find(j)+len(j)] +'[END]')

        print("="*92)
count = 0

for i,j, s in zip(train.text, train.selected_text, train.sentiment):

    if len(remove_html_char_ref(i))<len(j):

        count += 1

count
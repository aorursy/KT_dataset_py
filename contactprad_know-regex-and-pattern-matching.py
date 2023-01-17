# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

''' this kernel is to understand and help others understang different operator of Regex and how 

they could work together in pattern matching while performing text mining'''



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

sentence = '''Professor 12 Professr Professer Prod Tan '''

sentence1 = '''Professor Tan Eng Chye, NUS Deputy President and Provost, and Professor 

Menahem Ben-Sasson, President of HUJ signed the joint degree agreement at NUS, 

in the presence of Ambassador of Israel to Singapore Her Excellency Amira Arnon 

and about 30 invited guests, on Sept 25, 2013.'''



myRegExOperator = {

  'operator': 

      {'.':'Match any Single Character',

       '^Professor': 'Match the empty String that occurrs at the beginning of the line',

       '^Tan': 'Shoundnt give any result',

       'p^': 'dont know the outcome. this is incorrect usage it seems. shoulndt be any',

       'Tan$': 'Match the empty String that occurrs at the end of the line',

       'Professor$': 'Shoundn\'t give any result',

       '\d': 'Match any single digit',

       '\D': 'Match any non-digit character',

       '\w' : 'Match any alphanumeric character',

       'Professo?r': 'Match the preceding character 0 or 1 time',

       'Prof*': 'Zero or more of the preceding character',

       '[A-Z]': 'Match anything inside the square brackets for one character position once',

      '[^0-9]': 'Match any character excluding those in the square brackets for one character position once',

       '[A-Z]{1}': 'Match the preceding character, or character range n times',

       '[A-Z][a-z]{1,4}': '{n,m} Match the preceding character at least n times, but not more than m times',

       '(Professor)' : 'Explicit capture of the group',

       '(Profess(e|o)r)':'Separate two alternative values Professor or Professer',

       'Prof\\b': 'Matches empty string. to indicate word boundary. output will be None'

      } 

}



# Any results you write to the current directory are saved as output.
import re

def applyRegex(grammar):

    #cp = nltk.RegexpParser(grammar)

    regexp = re.compile(grammar)

    return regexp
for key in myRegExOperator['operator']:

    regexp = applyRegex(key)

    print("Key = ",key, '\n' , 'Description:' , 

          myRegExOperator['operator'][key] , "\n" , 

          "Produced output:", regexp.findall(sentence))
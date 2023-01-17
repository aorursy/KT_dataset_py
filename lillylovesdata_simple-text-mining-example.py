# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Load library
import pandas as pd
import numpy as np
import re
from collections import Counter
# For dispaly purpose
class display(object):
    """Display HTML representation of multiple objects"""
    template = """<div style="float: left; padding: 10px;">
    <p style='font-family:"Courier New", Courier, monospace'>{0}</p>{1}
    </div>"""
    def __init__(self, *args):
        self.args = args
        
    def _repr_html_(self):
        return '\n'.join(self.template.format(a, eval(a)._repr_html_())
                         for a in self.args)
    
    def __repr__(self):
        return '\n\n'.join(a + '\n' + repr(eval(a))
                           for a in self.args)
df = pd.read_csv('../input/consumer_complaints.csv')
# take a look the first five observation in the dataset
df.head()
print('Complain data set shape: ', df.shape)
df[(df['consumer_complaint_narrative'].notnull())]['consumer_complaint_narrative'].head()
# Subset the data by company
EQU = df[(df['consumer_complaint_narrative'].notnull())&(df['company']=='Equifax')]
EXP = df[(df['consumer_complaint_narrative'].notnull())&(df['company']=='Experian')]
TRU = df[(df['consumer_complaint_narrative'].notnull())&(df['company']=='TransUnion Intermediate Holdings, Inc.')]
TOTAL = df[(df['consumer_complaint_narrative'].notnull())]

# Take a look how many complaint related to each company
print(len(EQU),'complaints related to Equifax')
print(len(EXP),'complaints related to Experian')
print(len(TRU),'complaints related to TransUnion')
print(len(TOTAL),'complaints in Total')
# Create a empty Counter Object for the next step counting
EQU_counts = Counter()
EXP_counts = Counter()
TRU_counts = Counter()
TOTAL_counts = Counter()

EQU_lt = EQU['consumer_complaint_narrative'].tolist()
EXP_lt = EXP['consumer_complaint_narrative'].tolist()
TRU_lt = TRU['consumer_complaint_narrative'].tolist()
TOTAL_lt = TOTAL['consumer_complaint_narrative'].tolist()
#loop over all the words in the complaints and add up the counts
def count_word(complaints,word_counts):
    for i in range(len(complaints)):
        for word in re.split(r'\W+',  complaints[i]):
            word_counts[word] +=1
# count the word for each company's complaint lists
count_word(EQU_lt,EQU_counts)
count_word(EXP_lt,EXP_counts)
count_word(TRU_lt,TRU_counts)
count_word(TOTAL_lt,TOTAL_counts)

# extract the most common 10 words used in each company's complaint
EQU_counts_10 = EQU_counts.most_common(10)
EXP_counts_10 = EXP_counts.most_common(10)
TRU_counts_10 = TRU_counts.most_common(10)
TOTAL_counts_10 = TOTAL_counts.most_common(10)


# convert to dataframe for display
EQU_df = pd.DataFrame({'most 10 common (EQU)':EQU_counts_10})
EXP_df = pd.DataFrame({'most 10 common (EXP)':EXP_counts_10})
TRU_df = pd.DataFrame({'most  10 common (TRU)':TRU_counts_10})
Total_df = pd.DataFrame({'most 10 common (Total)':TOTAL_counts_10})

display('EQU_df', 'EXP_df', 'TRU_df', 'Total_df')
def calculate_ratio(word_counts,ratios):
    for word in list(word_counts):
        ratio = word_counts[word] / float(TOTAL_counts[word]+1)
        ratios[word] = ratio
# Again, create Counter object for ratio calculation
EQU_ratios = Counter()
EXP_ratios = Counter()
TRU_ratios = Counter()

# calculate the ratio for each company's complaint words
calculate_ratio(EQU_counts,EQU_ratios)
calculate_ratio(EXP_counts,EXP_ratios)
calculate_ratio(TRU_counts,TRU_ratios)


# words with the highest ratio 
EQU_df = pd.DataFrame({'most_common (EQU)':EQU_ratios.most_common(10)})
EXP_df = pd.DataFrame({'most_common (EXP)':EXP_ratios.most_common(10)})
TRU_df = pd.DataFrame({'most_common (TRU)':TRU_ratios.most_common(10)})

display('EQU_df', 'EXP_df', 'TRU_df')
# illustrate how the fuzzywuzzy.process work
from fuzzywuzzy import process

misspelled1 = 'Exquifaax'
misspelled2 = 'Exclude'
match = ['equifax']
fuzzy_score1 = process.extract(misspelled1, match)
fuzzy_score2 = process.extract(misspelled2, match)
print(fuzzy_score1)
print(fuzzy_score2)
#loop over all the words in the complaints and add up the counts
def count_word_new(word_lt,word_cnt,c_name_int,c_name):
    for i in range(len(word_lt)):
        lt = filter(None, re.split(r'\W+',  word_lt[i]))
        for word in lt:
            if word.lower().find(c_name_int) != -1:
                fuzzy_score = process.extract(word, c_name)[0][1]
                if fuzzy_score>=80:
                     continue
                else:
                    word_cnt[word] += 1

            else:
                word_cnt[word] += 1
# Create a empty Counter Object for the next step counting
EQU_counts2 = Counter()
EXP_counts2 = Counter()
TRU_counts2 = Counter()

# Again, create Counter object for ratio calculation
EQU_ratios2 = Counter()
EXP_ratios2 = Counter()
TRU_ratios2 = Counter()

# use the count_word_new to count
count_word_new(EQU_lt,EQU_counts2,"eq",["Equifax"])
count_word_new(EXP_lt,EXP_counts2,"expe",["Experian"])
count_word_new(TRU_lt,TRU_counts2,"tran",["TransUnion"])

calculate_ratio(EQU_counts2,EQU_ratios2)
calculate_ratio(EXP_counts2,EXP_ratios2)
calculate_ratio(TRU_counts2,TRU_ratios2)

EQU_df = pd.DataFrame({'most_common (EQU)':EQU_ratios2.most_common(10)})
EXP_df = pd.DataFrame({'most_common (EXP)':EXP_ratios2.most_common(10)})
TRU_df = pd.DataFrame({'most_common (TRU)':TRU_ratios2.most_common(10)})

display('EQU_df', 'EXP_df', 'TRU_df')

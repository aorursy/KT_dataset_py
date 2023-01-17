# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

from tqdm import tqdm

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
word_df = pd.read_csv('/kaggle/input/kernel205a192fe1/new_word.csv')

word_df.head()
#使用tag==1的中位数填充

word_df['c_p_max'] = word_df['c_p_max'].fillna(3.491680)

word_df['c_p_min'] = word_df['c_p_min'].fillna(3.272351)
a = []

for i in word_df['IE_L']:

    try:

        a.append(float(i))

    except:

        a.append(0)

word_df['IE_L'] = a



a = []

for i in word_df['IE_R']:

    try:

        a.append(float(i))

    except:

        a.append(0)

word_df['IE_R'] = a
for column in word_df.columns[2:]:

    print(word_df[column].describe())

    print(word_df[word_df['tag']==1][column].describe())
# count_ = 35

# iel_ =  2.5

# ier_ = 2.6

# avl_ = 5.0

# avr_ = 11.0

# cpmax_ = 13.0

# cpmin_ = 16.0



# tag_len = len(word_df[(word_df['count']>count_)&

#                       (word_df['IE_L']>iel_)&

#                       (word_df['IE_R']>ier_)&

#                       (word_df['AV_L']>avl_)&

#                       (word_df['AV_R']>avr_)&

#                       (word_df['c_p_max']<cpmax_)&

#                       (word_df['c_p_min']<cpmin_)]['tag'])

# tag_sum = sum(word_df[(word_df['count']>count_)&

#                       (word_df['IE_L']>iel_)&

#                       (word_df['IE_R']>ier_)&

#                       (word_df['AV_L']>avl_)&

#                       (word_df['AV_R']>avr_)&

#                       (word_df['c_p_max']<cpmax_)&

#                       (word_df['c_p_min']<cpmin_)]['tag'])

# print(tag_len,tag_sum)
count_ = 35

iel_ =  2.5

ier_ = 2.6

avl_ = 5.0

avr_ = 11.0

cpmax_ = 13.0

cpmin_ = 16.0

new_word_df = word_df[(word_df['count']>count_)&

                      (word_df['IE_L']>iel_)&

                      (word_df['IE_R']>ier_)&

                      (word_df['AV_L']>avl_)&

                      (word_df['AV_R']>avr_)&

                      (word_df['c_p_max']<cpmax_)&

                      (word_df['c_p_min']<cpmin_)]
new_word_df.to_csv('new_word.csv')
# def merge(begain,end,num,float_=False):

#     if float_ == False:

#         a = np.unique(np.linspace(begain,end,num).astype(int))

#     else:

#         a = np.linspace(begain,end,num)

#     return a



# count = merge(1,35,12)

# IE_L = merge(0,2.5,3,float_=True)  #>

# IE_R = merge(0,2.6,3,float_=True)  #>

# AV_L = merge(0,11,5) 

# AV_R = merge(0,11,5) 

# c_p_max = merge(7,16,4) 

# c_p_min = merge(7,16,4)
# results = np.array([0,0,0,0,0,0,0,0,0])

# for count_ in tqdm(count):

#     for iel_ in IE_L:

#         for ier_ in IE_R:

#             for avl_ in AV_L:

#                 for avr_ in AV_R:

#                     for cpmax_ in c_p_max:

#                         for cpmin_ in c_p_min:

#                             tag_len = len(word_df[(word_df['count']>count_)&

#                                                   (word_df['IE_L']>iel_)&

#                                                   (word_df['IE_R']>ier_)&

#                                                   (word_df['AV_L']>avl_)&

#                                                   (word_df['AV_R']>avr_)&

#                                                   (word_df['c_p_max']<cpmax_)&

#                                                   (word_df['c_p_min']<cpmin_)]['tag'])

#                             tag_sum = sum(word_df[(word_df['count']>count_)&

#                                                   (word_df['IE_L']>iel_)&

#                                                   (word_df['IE_R']>ier_)&

#                                                   (word_df['AV_L']>avl_)&

#                                                   (word_df['AV_R']>avr_)&

#                                                   (word_df['c_p_max']<cpmax_)&

#                                                   (word_df['c_p_min']<cpmin_)]['tag'])

#                             param = np.array([count_,iel_,ier_,avl_,avr_,cpmax_,cpmin_,tag_len,tag_sum])

#                             results = np.c_[results,param]
# count = [0,1,2,3,4,5]  # >

# IE_L = [0,0.2,0.4,0.6,0.8,1]  #>

# IE_R = [0,0.2,0.4,0.6,0.8,1]  #>

# AV_L = [0,1,2] #>

# AV_R = [0,1,2] #>

# c_p_max = [16,12,8,6,5] # <

# c_p_min = [16,12,8,6,5] # <
# result_df = pd.DataFrame(results.T,columns=['count','ie_l','ie_r','av_l','av_r','c_p_max','c_p_min','prepare_count','count_in_dict'])

# result_df.head()
# result_df.to_csv('result.csv')
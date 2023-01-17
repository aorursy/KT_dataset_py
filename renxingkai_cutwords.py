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
file_related="/kaggle/input/clean-irdata/clean_data/news1000_clean.csv"

file_unrelated="/kaggle/input/clean-irdata/clean_data/news_unrelated_clean.csv"

file_whole="/kaggle/input/wholedata/datawhole_clean.csv"
data_related=pd.read_csv(file_whole,encoding='utf-8', lineterminator="\n")
data_related.columns=['NAME', 'CONTENT', 'STATUS', 'name_content', 'name_content_clean']
data_related.head()
import jieba



# 创建停用词列表

def stopwordslist():

    stopwords = [line.strip() for line in open("/kaggle/input/stopwords/zhstops.txt",encoding='UTF-8').readlines()]

    return stopwords



# 对句子进行中文分词

def seg_depart(sentence):

    # 对文档中的每一行进行中文分词

    sentence_depart = jieba.cut(sentence.strip())

    # 创建一个停用词列表

    stopwords = stopwordslist()

    # 输出结果为outstr

    outstr = ''

    # 去停用词

    for word in sentence_depart:

        if word not in stopwords:

            if word != '\t':

                outstr += word

                outstr += " "

    return outstr
seg_depart("我能打几级了")
import numpy as np

import pandas as pd

from tqdm import tqdm,tqdm_notebook

tqdm.pandas(desc="my")
data_related['name_content_clean_cut']=data_related['name_content_clean'].progress_apply(seg_depart)
import jieba.analyse

import jieba
def textrank(text,topK=30):

    kw_textrank=jieba.analyse.textrank(text,topK=topK)

    return kw_textrank
data_related['textrank']=data_related['name_content_clean_cut'].progress_apply(textrank)
data_related.to_csv("wholedata_textrank.csv",encoding='utf-8',index=False)
# data_related.to_csv("n10000_cut.csv",encoding='utf-8',index=False)
# data_unrelated=pd.read_csv(file_unrelated,encoding='utf-8')
# data_unrelated.head()
# data_unrelated['name_content_clean_cut']=data_unrelated['name_content_clean'].progress_apply(seg_depart)
# data_unrelated.to_csv("n288_cut.csv",encoding='utf-8',index=False)
# from collections import Counter





# def precision_recall_f1(prediction, ground_truth):

#     """

#     This function calculates and returns the precision, recall and f1-score

#     Args:

#         prediction: prediction string or list to be matched

#         ground_truth: golden string or list reference

#     Returns:

#         floats of (p, r, f1)

#     Raises:

#         None

#     """

#     if not isinstance(prediction, list):

#         prediction_tokens = prediction.split()

#     else:

#         prediction_tokens = prediction

#     if not isinstance(ground_truth, list):

#         ground_truth_tokens = ground_truth.split()

#     else:

#         ground_truth_tokens = ground_truth

#     common = Counter(prediction_tokens) & Counter(ground_truth_tokens)

#     num_same = sum(common.values())

#     if num_same == 0:

#         return 0, 0, 0

#     p = 1.0 * num_same / len(prediction_tokens)

#     r = 1.0 * num_same / len(ground_truth_tokens)

#     f1 = (2 * p * r) / (p + r)

#     return p, r, f1

# keywords="益阳市|桃江县|安化县|南县|资阳市|赫山区|(益阳市+沅江市)|大通湖区|沅江市|益阳|桃江|安化|南县|资阳|赫山|(益阳+沅江)|大通湖|沅江"
# keywords_split=keywords.replace("|"," ").replace("+","").replace("(","").replace(")","")
# keywords_split
# precision_recall_f1(data_unrelated['name_content_clean_cut'].iloc[0],keywords_split)
# data_unrelated['precision']=data_unrelated['name_content_clean_cut'].apply(precision_recall_f1, args=(word_list,))
# data_unrelated['precision']=""

# data_unrelated['recall']=""

# data_unrelated['f1']=""



# for i in tqdm(range(len(data_unrelated))):

#     p,r,f1=precision_recall_f1(data_unrelated['name_content_clean_cut'][i],keywords_split)

#     data_unrelated['precision'][i]=p

#     data_unrelated['recall'][i]=r

#     data_unrelated['f1'][i]=f1
# data_unrelated.to_csv("n288_prf1.csv",encoding='utf-8',index=False)
# data_related['precision']=""

# data_related['recall']=""

# data_related['f1']=""



# for i in tqdm(range(len(data_related))):

#     p,r,f1=precision_recall_f1(data_related['name_content_clean_cut'][i],keywords_split)

#     data_related['precision'][i]=p

#     data_related['recall'][i]=r

#     data_related['f1'][i]=f1
# data_related.to_csv("n1000_prf1.csv",index=False,encoding='utf-8')
# data_related['STATUS'].value_counts()
# data_related.head(50)
# np.mean(data_unrelated['recall'])
# import seaborn as sns
# sns.kdeplot(data_unrelated['recall'])
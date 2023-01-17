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
from nltk.book import *
text1
text1.concordance('monstrous')
#寻找与monstrous相似的上下文
text1.similar('monstrous')
#寻找两个共同的上下文
text2.common_contexts(['monstrous','very'])
#我们也可以判断词在文本中的位置：从文本开头算起在它前面有多少词。
#这个位置信息可以用离散图表示。每一个竖线代表一个单词，每一行代表整个文本
text4.dispersion_plot(['citizens','democracy','freedom','duties','America'])
#随机文本
text3.generate()
len(text3)
sorted(set(text3))
len(set(text3))
len(set(text3))/len(text3)
#计算特定词在文本中的次数，与百分比
text3.count('smote'),100 * text4.count('a') / len(text4)
text4[173]
#找出一个词第一次出现的索引
text4.index('awaken')
#切片
text5[16715:16735]
saying=['After','all','is','said','and','done',
       'more','is','said','than','done']
tokens=set(saying)
tokens=sorted(tokens)
tokens[-2:]
fdist1=FreqDist(text1)
print(fdist1)
fdist1.most_common(50)
##词汇的累积频率图
fdist1.plot(50,cumulative=True)
#查看低频词汇(只出现了一次的词（所谓的hapaxes）)
fdist1.hapaxes(),len(fdist1.hapaxes())
V=set(text1)
long_words=[w for w in V if len(w)>15]
sorted(long_words)
#天语料库中所有长度超过7 个字符，且出现次数超过7 次的词
fdist5=FreqDist(text5)
sorted(w for w in set(text5) if len(w)>7 and fdist5[w]>7)
list(bigrams(['more', 'is', 'said', 'than', 'done']))
#基于单个词的频率预期得到的更频繁出现的双连词
text4.collocations()
[len(w) for w in text1]
fdist=FreqDist(len(w) for w in text1)
print(fdist)
fdist
fdist.most_common()
fdist.max()
fdist[3]
fdist.freq(3)
sorted(w for w in set(text1) if w.endswith('ableness'))
sorted(term for term in set(text4) if 'gnt' in term)
sorted(item for item in set(text6) if item.istitle())
sorted(item for item in set(text1) if item.isdigit())
sorted(w for w in set(text1) if not w.islower())
[w.upper() for w in text1]
len(set(word.lower() for word in text1 if word.isalpha()))

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
!pip install nltk
from nltk.book import *
text1
# 输出词频分布图
text4.dispersion_plot(["freedom", "citizens"])
# 计算某个单词出现的次数
text3.count("smote")
# 在text5中，单词"lol"出现了多少次？
text5.count("lol")
sent2
# 连接两个句子（列表）
sent1 + sent2
# 从text对象中切片出列表
text5[4545: 4550]
fdist1 = FreqDist(text1)
fdist1
# 返回值是一个字典，键值对即为“词”和“词频”
list(fdist1.keys())[:20]  # 查看前20个key
# 输出词频的统计图
fdist1.plot(50, cumulative=True)  # 累积模式，会将词频从左到右逐个
fdist1.plot(50)  # 词频方式，参数50表示只显示前五十个词
# 使用 hapaxes 方法,查看"低频词"
# 方法的调用者是"词频字典"
# 方法返回一个列表
words = fdist1.hapaxes()
words[:20]  # 查看列表中前20个词
# 低频词（只出现一次的词）的个数也相当多
len(fdist1.hapaxes())
v = set(text1)  # 去重
long_words = [w for w in v if len(w) > 15]  # 使用列表生成式筛选
sorted(long_words)
# 同样是上面的“15词长”规则，看看在"网络文本"语料text5中能发现什么
v = set(text5)
words = [w for w in v if len(w) > 15]
sorted(words)[:20]  # 查看前20个词
fdist5 = FreqDist(text5)  # 提取text5中所有词的词频
# 选出那些长度大于7，词频也大于7的词
words = [w for w in set(fdist5) if len(w) > 7 and fdist5[w] > 7]
sorted(words)
# 找出text1中频繁的双连词
text1.collocations()
# 试试text5?
text5.collocations()
# 其他功能：生成词长的分布
fdist1 = FreqDist(text1)
word_len_dist = {}
for w in fdist1.keys():
    word_len_dist[len(w)] = word_len_dist.get(len(w), 0) + fdist1[w]
# 观察图表可以发现长度为3的词在这个文本中是最多的
FreqDist(word_len_dist).plot()
# 从text1的词语中选出"ableness"结尾的词
sorted([w for w in set(text1) if w.endswith("ableness")])
import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt # 画图常用库

vect = CountVectorizer()

example = ['I love you, good bad bad', 'you are soo good']



result = vect.fit_transform(example)

print(result)

print (vect.vocabulary_)



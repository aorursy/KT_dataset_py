# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))
data = pd.read_csv("../input/9gag.csv")
print(data.columns)
# Any results you write to the current directory are saved as output.
def convert_to_list(string):
    to_list = string.strip('][').lower().split(',')
    return to_list
tag_score = {}
tags = []
for i in range(data.shape[0]):
    item = data["all_tags"][i]
    score = data["upVoteCount"][i] + data["commentsCount"][i] * 2
    if item:
        for x in convert_to_list(item):
            x = x.lstrip().rstrip()
            if x:
                if (x not in tag_score):
                    tag_score[x] = score
    #                 tags.append(x.lstrip().rstrip())
                else:
                    tag_score[x] += score
# print(tag_score)
key = max(tag_score, key=lambda key: tag_score[key])
print(key, tag_score[key])
# print(tag_score)

score_keys = sorted(tag_score, key=(lambda key:tag_score[key]), reverse=True)[:10]
print(score_keys)
score_values = []
for key in score_keys:
    score_values.append(tag_score[key])
print(score_values)


plt.bar(np.arange(len(score_keys)), score_values, align='center', alpha=0.5)
# index = np.arange(len(label))
#     plt.bar(index, no_movies)
plt.xlabel('Category', fontsize=10)
plt.ylabel('Score', fontsize=10)
plt.xticks(np.arange(len(score_keys)), score_keys, fontsize=10, rotation=30)
plt.title('Most liked joke type')
 
plt.show()
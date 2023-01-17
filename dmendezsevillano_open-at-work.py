# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
answer=["Always open", "selectively open","hide LGBT identity"]

ques='open_at_work'

#subset='Gay'

country='Average' 

data_0=data[data.CountryCode==country][data.answer == answer[0]][data.question_code==ques].percentage

data_1=data[data.CountryCode==country][data.answer == answer[1]][data.question_code==ques].percentage

data_2=data[data.CountryCode==country][data.answer == answer[2]][data.question_code==ques].percentage

labels=list(data[data.CountryCode==country][data.answer == answer[0]][data.question_code==ques].subset)

x=np.arange(len(labels))

width=0.5

sum_list=[]

for (item1, item2) in zip(list(data_0), list(data_1)):

    sum_list.append(item1+item2)

fig, ax = plt.subplots(figsize=(13.5, 12), dpi= 80, facecolor='w', edgecolor='k')

rects1 = ax.bar(x, data_0, width, label=answer[0])

rects2 = ax.bar(x, data_1, width, bottom=data_0, label=answer[1])

rects3 = ax.bar(x, data_2, width, bottom=sum_list, label=answer[2])

ax.set_ylabel('Percentage (%)')

ax.set_title(country+' '+ques)

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation=40, ha='right')

ax.legend(answer)

plt.show()
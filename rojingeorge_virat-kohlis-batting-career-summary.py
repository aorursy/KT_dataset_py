# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
data_Score = pd.read_csv('../input/kohli_score_sheet_v1.csv')
#print(data_Score['Type'])
no_of_match = data_Score['M']
type_of_match = data_Score['Type']
row=4
#print(os.listdir("../input/"))

#create graph 
plt.ylabel('No of matches',color = (0.3,0.1,0.4,0.6))
plt.title('Kohlis career summary',color =(0.3,0.1,0.4,0.6))
plt.xlabel('Type of match',color = (0.3,0.1,0.4,0.6))

objects = list(data_Score['Type'])
y_pos = np.arange(len(objects))
performance = list(data_Score['M'])
avg_score = list(data_Score['Avg'])
plt.bar(y_pos, performance, align='center', color = (0.3,0.5,0.4,0.6), width =0.5)
plt.xticks(y_pos, objects)

#assign avg scored in to each bar.
for i in range(4):
   plt.text(x=i, y=no_of_match[i], s=str('avg = '+str(avg_score[i])))
# Text on the top of each barplot

#plt.legend()
# 
plt.show()

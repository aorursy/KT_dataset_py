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
import matplotlib.pyplot as plt

data = {'apples': 10, 'oranges': 15, 'lemons': 5, 'limes': 20}
names = list(data.keys())
values = list(data.values())

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
axs[0].bar(names, values)
axs[1].scatter(names, values)
axs[2].plot(names, values)
fig.suptitle('Categorical Plotting')
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
A = np.arange(1,5)
B = A**2
C = A**3
fig1, ax1=plt.subplots(1,2)
ax1[0].plot(A, B)
ax1[1].plot(B, A)
# fig, ax = plt.subplots(figsize=(14,7))
fig, ax = plt.subplots(2,3)
# ax[0].***
# ax[1].***
ax[0][0].plot(A,B)
ax[0][1].plot(B,A)
ax[0][2].plot(A,B)
ax[1][0].plot(A,B)
ax[1][1].plot(B,B)
ax[1][2].plot(A,B)
# plt.show()
fig2, ax2 = plt.subplots(figsize=(14,7))
ax2.plot(A,B,label='A-B')
ax2.plot(B,A,label='B-A')
ax2.legend()
ax2.set_title('Title',fontsize=18)
ax2.set_xlabel('xlabel', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
ax2.set_ylabel('ylabel', fontsize='x-large',fontstyle='oblique')
plt.show()
# ax2.set_aspect('equal') 
# ax2.minorticks_on() 
# ax2.set_xlim(0,16) 
# ax2.grid(which='minor', axis='both')
# ax2.xaxis.set_tick_params(rotation=45,labelsize=18,colors='w') 
# start, end = ax2.get_xlim() 
# ax2.xaxis.set_ticks(np.arange(start, end,1)) 
# ax2.yaxis.tick_right()
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
symptoms={'symptom':['Fever',
        'Dry cough',
        'Fatigue',
        'Sputum production',
        'Shortness of breath',
        'Muscle pain',
        'Sore throat',
        'Headache',
        'Chills',
        'Nausea or vomiting',
        'Nasal congestion',
        'Diarrhoea',
        'Haemoptysis',
        'Conjunctival congestion'],'percentage':[87.9,67.7,38.1,33.4,18.6,14.8,13.9,13.6,11.4,5.0,4.8,3.7,0.9,0.8]}

symptoms=pd.DataFrame(data=symptoms,index=range(14))
symptoms
from wordcloud import WordCloud, ImageColorGenerator
text = " ".join(str(each) for each in symptoms.symptom)
wordcloud = WordCloud(max_words=200,colormap='Set3', background_color="white").generate(text)
plt.figure(figsize=(10,6))
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='Bilinear')
plt.axis("off")
plt.figure(1,figsize=(12, 12))
plt.show()

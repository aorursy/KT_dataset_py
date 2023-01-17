# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

skills_df = pd.read_csv('../input/job_skills.csv')

print(skills_df.columns)
print(skills_df.head())
from collections import Counter

print(skills_df.loc[:,'Company'].unique())

company_counter=Counter()

for ent in skills_df.loc[:, 'Company']:
    if ent in company_counter:
        company_counter[ent] += 1
    else:
        company_counter[ent] = 1
print(company_counter)

stopwords = nltk.corpus.stopwords.words('english')
pref_qual_text = skills_df['Preferred Qualifications'].str.lower() #first lower case text
pref_qual_text = pref_qual_text.str.replace(r'[^\w\s]+', '')
pref_qual_text = pref_qual_text.str.cat(sep=' ') # join all the preferred qualifications entries

pref_qual_words = nltk.word_tokenize(pref_qual_text) #stopwords

pref_qual_words_dist = nltk.FreqDist(pref_qual_words)#with stopwords

pref_qual_words_except_stop_dist = nltk.FreqDist(w for w in pref_qual_words if w not in stopwords)#without stopwords

top_20 = pref_qual_words_except_stop_dist.most_common(20)

labels, values = zip(*top_20)
indexes = np.arange(len(labels))
width = 1

plt.bar(indexes, values)
plt.xticks(indexes + width * 0.5-0.5, labels,rotation=90)
plt.xlabel("Term")
plt.ylabel('Count')
plt.title('Count of Top 20 Preferred Qualifications')
plt.show()



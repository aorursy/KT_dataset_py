# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/questions.csv').dropna() #There is a nan in here?
corpus =pd.concat([df.question1,df.question2])



cv = CountVectorizer().fit(corpus)



duplicates = df.loc[df.is_duplicate==1,['question1','question2']]

nondupes = df.loc[df.is_duplicate==0,['question1','question2']]

non_dupes_csim = []



for j in range(nondupes.shape[0]):

    

    a = cv.transform([nondupes.iloc[j].question1])

    b = cv.transform([nondupes.iloc[j].question2])

    

    non_dupes_csim.append( cosine_similarity(a,b).ravel()[0])

duplicates_csim = []



for j in range(duplicates.shape[0]):

    

    a = cv.transform([duplicates.iloc[j].question1])

    b = cv.transform([duplicates.iloc[j].question2])

    

    duplicates_csim.append( cosine_similarity(a,b).ravel()[0])
plt.hist(duplicates_csim, alpha = 0.5, color = 'r', normed = True, bins = np.linspace(0,1,11))

plt.hist(non_dupes_csim, alpha = 0.5, color = 'b', normed = True, bins = np.linspace(0,1,11))
np.mean(non_dupes_csim)
np.mean(duplicates_csim)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import string
import numpy as np
import matplotlib.pyplot as plt
print(string.punctuation)
stopwords_en = pd.read_csv('/kaggle/input/dataset/english.txt')
print(stopwords_en.head())

title = 'Studying customer loyalty: An analysis for an online retailer in Turkey'

#Convert the text to lower case, remove punctuation and stop words
print(title)

title = title.lower()
print(title)

title = title.translate(string.punctuation)
print(title)


title = [word for word in title.split() if word not in stopwords_en.values ]

print(title)
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Load the data on publishing of Turkish academics. 
df_paper = pd.read_csv('/kaggle/input/dataset/papers.csv')
df_keywords = pd.read_csv('/kaggle/input/dataset/paperKeywords.csv')
df_field = pd.read_csv('/kaggle/input/dataset/fieldofStudy.csv')
df_affiliation = pd.read_csv('/kaggle/input/dataset/Affiliations.csv')

print(df_paper.head())
print(df_keywords.head())
print(df_field.head())
print(df_affiliation.head())
#Check how many papers use the word 'data' on their titles since 1997 and do the same for word 'science'.

df_data = df_paper[df_paper['titleClean'].str.contains('data')]
years_data,paper_in_year_data = np.unique(df_data.year, return_counts=True)

f, (ax1, ax2) = plt.subplots(1, 2, figsize = (15,5), sharey=True)
ax1.plot(paper_in_year_data[-20:])
plt.sca(ax1)
plt.xticks(range(20), years_data[-20:], rotation = 'vertical')
ax1.set_title('Number of papers with the word \'data\' on the title')
plt.sca(ax2)

df_science = df_paper[df_paper['titleClean'].str.contains('science')]
years_science,paper_in_year_science = np.unique(df_science.year, return_counts=True)

ax2.plot(paper_in_year_science[-20:])
plt.xticks(range(20), years_science[-20:], rotation = 'vertical')
ax2.set_title('Number of papers with the word \'science\' on the title')
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Create tf-idf matrix
df_paper1 = df_paper.iloc[0:1000,:]
tdm = TfidfVectorizer(min_df=20, stop_words='english')
paper_term_document = tdm.fit_transform(df_paper1['titleClean'])
print(paper_term_document)
print(tdm.vocabulary_)

#Create count vectorizer matrix
cv = CountVectorizer(min_df=20, stop_words='english')
paper_term_count_document = cv.fit_transform(df_paper1['titleClean'])
print(paper_term_count_document)
print(cv.vocabulary_)
#To find similarity between documents, kmeans algorithm applied.
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 60, n_init = 10, random_state=1)

kmeans.fit(paper_term_document)

labels = kmeans.predict(paper_term_document)

pd.options.display.max_colwidth = 120
pd.options.display.max_rows = 200
print("%1000s" % df_paper1.iloc[labels==labels[0],:]['titleClean'])
#To find how words are correlated with each other, correlation matrix have been calculated

df_ptd = pd.DataFrame(paper_term_count_document.todense())

vocabulary = dict((v, k) for k, v in cv.vocabulary_.items())
df_ptd.columns=vocabulary.items()
corr_mat = df_ptd.corr()
import seaborn as sns
sns.set(context="paper", font="arial", style = "whitegrid", font_scale=2.0)


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(15, 15))

# Draw the heatmap using seaborn
sns.heatmap(corr_mat, square=True, cmap = 'YlGnBu', xticklabels=df_ptd.columns, yticklabels=df_ptd.columns)
f.tight_layout()
plt.title('Correlation Map')
plt.legend(fontsize=12)
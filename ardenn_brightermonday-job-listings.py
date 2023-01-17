import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import string
jobs=pd.read_csv("../input/jobs.csv")
jobs.head()
jobs.describe()
jobs.nunique()
jobs.isnull().any()
jobs=jobs.astype({"category":"category","job_type":"category","location":"category"})
#Jobs by category

jobs_by_category=jobs["category"].value_counts(normalize=True).sort_values(ascending=False)

jobs_by_category.round(3).mul(100).plot(kind="barh",figsize=[16,10])
#Jobs by Location

jobs_by_location=jobs["location"].value_counts(normalize=True).sort_values(ascending=False)

jobs_by_location.round(3).mul(100).plot(kind="barh",figsize=[10,8], rot=0)
#Job by salary

# jobs_by_salary=jobs["salary"].value_counts(normalize=True).sort_values(ascending=False)

# jobs_by_salary.round(3).mul(100)
#Top Employers

top_employers=jobs["employer"].value_counts(normalize=True)

top_employers.sort_values(ascending=False).round(3).mul(100)[:15].plot(kind="barh",figsize=[10,6])
#Find most sought after skills

# all_words=nltk.word_tokenize(" ".join(jobs["description"].str.lower()))

# stopwords = nltk.corpus.stopwords.words('english')+list(string.punctuation)

# clean_words = [wd for wd in all_words if wd not in stopwords]
#plot the frequency of the words used

pd.Series(nltk.FreqDist(clean_words)).sort_values(ascending=False)[:50].plot(kind="barh",figsize=[16,10])
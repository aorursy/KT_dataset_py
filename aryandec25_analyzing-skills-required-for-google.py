import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir('../input/'))
data=pd.read_csv('../input/google-job-skills/job_skills.csv')

data.head()
data.info()
data.describe()
sns.set(style="darkgrid")

sns.countplot(data['Company'])

plt.title('')



print(data['Company'].value_counts())
plt.title('Top 10 Job Titles')

top_title=data['Title'].value_counts().head(10)

top_title.plot(kind='bar')
plt.title('Top 10 Location')

top_location=data['Location'].value_counts().sort_values(ascending=False).head(10)

top_location.plot(kind='bar')
plt.figure(figsize=(10,6))

data['Category'].value_counts().plot(kind='bar')
data.dropna(inplace=True)
data.isnull().any()
from collections import Counter

cnt = Counter()

for text in data['Minimum Qualifications'].values:

    for word in text.split():

        cnt[word] += 1

        

cnt.most_common(10)
for text in data['Preferred Qualifications'].values:

    for word in text.split():

        cnt[word] += 1

        

cnt.most_common(10)
import numpy as np

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
reports = pd.read_csv('/kaggle/input/2019-autonomous-vehicle-disengagement-reports/2019AutonomousVehicleDisengagementReports.csv')

reports_ftf = pd.read_csv('/kaggle/input/2019-autonomous-vehicle-disengagement-reports/2018-19_AutonomousVehicleDisengagementReports(firsttimefilers).csv')
reports.iloc[322,-1]
import nltk

from nltk.util import ngrams

from nltk.collocations import BigramCollocationFinder

from nltk.metrics import BigramAssocMeasures

import re
# This creates one long string to perform n-gram operations on

super_string = reports['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'].astype(str).sum()

super_string = re.sub(r'[^\w\s]','',super_string)

super_string = super_string.split(' ')

stop_words = set(nltk.corpus.stopwords.words('english'))

super_string = [word for word in super_string if word not in stop_words]

len(super_string)
word_fd = nltk.FreqDist(super_string)

monogram_fd = nltk.FreqDist(nltk.ngrams(super_string,1))

monogram_fd.most_common(9)
bigram_fd = nltk.FreqDist(nltk.bigrams(super_string))

bigram_fd.most_common(10)


bigram_fd = nltk.FreqDist(nltk.ngrams(super_string, 6))

bigram_fd.most_common(10)
skipgram_fd = nltk.FreqDist(nltk.skipgrams(super_string, n=4, k=3))



skipgram_fd.most_common(9)
import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
list(pd.value_counts(reports['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT']).iloc[:15].index)
figure(num=None, figsize=(24, 22))

sns.countplot(y='DESCRIPTION OF FACTS CAUSING DISENGAGEMENT', palette='plasma',data=reports,order=pd.value_counts(reports['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT']).iloc[:30].index)

sns.despine(bottom=True, left=True)

plt.xticks(rotation=90);
figure(num=None, figsize=(20, 12))

sns.countplot(y='DESCRIPTION OF FACTS CAUSING DISENGAGEMENT', palette='plasma',data=reports_ftf,order=pd.value_counts(reports_ftf['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT']).iloc[:30].index)

sns.despine(bottom=True, left=True)

plt.xticks(rotation=90);
reports.loc[reports['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'] == 'Safety Driver proactive disengagement.'].describe()
reports_ftf.loc[reports_ftf['DESCRIPTION OF FACTS CAUSING DISENGAGEMENT'] == 'Software Discrepancy'].describe()
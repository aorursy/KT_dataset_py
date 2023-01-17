# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
marathon = pd.read_csv("../input/marathon_results_2016.csv")

marathon.columns
marathon.head()
m_marathon = marathon[marathon['M/F'] == 'M'].sample(n = 1000)

f_marathon = marathon[marathon['M/F'] == 'F'].sample(n = 1000)

_m = plt.scatter(m_marathon['Age'],m_marathon['Overall'],label='Male');

_f = plt.scatter(f_marathon['Age'],f_marathon['Overall'],label='Female');

plt.legend(handles=[_m, _f])

plt.xlabel('Age')

plt.ylabel('Ranking')
valid_Bib = marathon['Bib'].apply(lambda x: x.isdigit())

marathon_validBib = marathon[valid_Bib]

BibIntSeries = [int(x) for x in marathon_validBib['Bib']]

marathon_validBib.Bib = BibIntSeries

m_marathon = marathon_validBib[marathon_validBib['M/F'] == 'M'].sample(n = 500)

f_marathon = marathon_validBib[marathon_validBib['M/F'] == 'F'].sample(n = 500)

_m = plt.scatter(m_marathon['Bib'],m_marathon['Gender'],label='Male');

_f = plt.scatter(f_marathon['Bib'],f_marathon['Gender'],label='Female');

plt.legend(handles=[_m, _f])

plt.xlabel('Bib')

plt.ylabel('Ranking')
marathon_gender = marathon.loc[:,['M/F','Overall']]

marathon_gender.boxplot(by=['M/F'])

plt.ylabel('Overall Ranking')

m_topCountryList = " "

m_marathon = marathon[marathon['M/F'] == 'M']

m_marathon_top = m_marathon.sort_values(by=['Gender']).head(100)



for i in m_marathon_top['Country']:

    m_topCountryList += (i+" ")

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(m_topCountryList)

plt.imshow(wordcloud)



f_topCountryList = " "

f_marathon = marathon[marathon['M/F'] == 'F']

f_marathon_top = f_marathon.sort_values(by=['Gender']).head(100)

for i in f_marathon_top['Country']:

    f_topCountryList += (i+" ")

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(f_topCountryList)

plt.imshow(wordcloud)
CountryList = ""

for i in marathon['Country']:

    CountryList += (i+" ")

wordcloud = WordCloud(stopwords=STOPWORDS,background_color='white').generate(CountryList)

plt.imshow(wordcloud)
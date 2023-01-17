# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cr = pd.read_csv('../input/conversionRates.csv',encoding="ISO-8859-1", low_memory=False)

ffr = pd.read_csv('../input/freeformResponses.csv',encoding="ISO-8859-1", low_memory=False)

mcr = pd.read_csv('../input/multipleChoiceResponses.csv',encoding="ISO-8859-1", low_memory=False)

schema = pd.read_csv('../input/schema.csv',encoding="ISO-8859-1", low_memory=False)
cr.head()
ffr.head()
mcr.head()
schema.head()
import seaborn as sns

import matplotlib.pyplot as plt

# plt.figure(figsize=(30,10))

sns.countplot(mcr['GenderSelect'],data = mcr)

plt.xticks(rotation = 90)

mcr.GenderSelect.value_counts()
plt.figure(figsize=(30,10))

sns.boxplot(x = mcr.Country,y = mcr.Age,data = mcr)

plt.xticks(rotation=90)

plt.show()
#in number formats

mcr.Country.value_counts()
# plt.figure(figsize=(20,10))

sns.boxplot(x = mcr.Age,y = mcr.GenderSelect,data = mcr)

# plt.xticks(rotation=90)

plt.show()
mcr.Age.describe()
mcr.Country.value_counts().sort_values(ascending=False).head(15)
plt.figure(figsize=(25,5))

sns.countplot(mcr.Country)

plt.xticks(rotation=90)

plt.show()
mcr.LanguageRecommendationSelect.value_counts().sort_values(ascending=False)
plt.figure(figsize=(20,5))

sns.countplot(mcr.LanguageRecommendationSelect)

plt.xticks(rotation = 90)

plt.show()
mcr.MLMethodNextYearSelect.value_counts()

mcr.MLToolNextYearSelect.value_counts().sort_values(ascending=False).head(10)
plt.figure(figsize=(20,5))

sns.countplot(mcr.MLToolNextYearSelect)

plt.xticks(rotation = 90)

plt.show()
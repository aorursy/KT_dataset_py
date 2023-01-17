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



%matplotlib inline

import matplotlib as mpl 

import matplotlib.pyplot as plt

plt.style.use('ggplot')

print ('Matplotlib version: ', mpl.__version__)

print ('Pandas version: ',pd.__version__)
# Analyze Q1 responses



path = '../input/kaggle-survey-2019/multiple_choice_responses.csv'

dfresp = pd.read_csv(path)



# Graph that shows response counts by age group



dfresptop = dfresp.Q1.value_counts().sort_index()

dfresptop.plot(kind='barh',color='Green')



for index, value in enumerate(dfresptop):

    label = format(int(value),)

    plt.annotate(label,xy=(value - 400, index - 0.20), color='white')

    

plt.show()
# Analyze Q2 responses

# Graph that shows responses by gender

dfresp.Q2.value_counts().sort_index().plot(kind='bar',color='green')
# Analyze Q3 responses

# Graph that shows Q3 respones by country with atleast 500 or more responses

country_counts = dfresp.Q3.value_counts()



# top countries that have atleast 500 respondents

country_counts[country_counts >= 500].count()



# select and show countries that have 500 or more respondents

top_countries = country_counts[country_counts >= 500].index

dfresp_top_countries =  dfresp[dfresp.Q3.isin(top_countries)]

dfresp_top_countries.Q3.value_counts().sort_values().plot(kind='barh',color='green')
# Analyze Q4 responses

dfresp['Q4'].value_counts().sort_values().plot(kind='barh',color='green')
# Analyze Q5 responses

dfresp['Q5'].value_counts().sort_values().plot(kind='barh',color='green')
# Analyze Q6 responses

# show the count of company sizes from the most to least 

dfresp['Q6'].value_counts().sort_values().plot(kind='barh',color='green')
# Analyze Q7 responses

# show the count of data science workload individuals

dfresp['Q7'].value_counts().sort_values().plot(kind='barh',color='green')
# Analyze Q8 responses

# Machine learning methods incorporated into business

dfresp['Q8'].value_counts().sort_values().plot(kind='barh',color='green')
# Analyze Q11 responses

dfresp['Q11'].value_counts().sort_values().plot(kind='barh',color='green')

# Analyze Q14 responses

dfresp['Q14'].value_counts().sort_values().plot(kind='barh',color='green')
# Analyze Q15 responses

dfresp['Q15'].value_counts().sort_values().plot(kind='barh',color='green')
path = '../input/kaggle-survey-2019/questions_only.csv'

dfq = pd.read_csv(path)

dfq.describe()

dfq.head()
path = '../input/kaggle-survey-2019/survey_schema.csv'

dfsch = pd.read_csv(path)

dfsch.describe()

dfsch.head(2)
path = '../input/kaggle-survey-2019/other_text_responses.csv'

dforesp = pd.read_csv(path)

dforesp.head()
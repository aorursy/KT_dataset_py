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
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.plotly as py
file = "../input/h1b_kaggle.csv"

data = pd.read_csv(file,sep=",")

data.head()
#h1b applied vs year

sns.set_context("notebook",font_scale=1.0)

plt.figure(figsize=(12,4))

plt.title('h1b petitions filed per year')

sns.countplot(data['YEAR'])
#rejection vs accepted h1b application

data['CASE_STATUS'].value_counts().plot(kind='barh')
#worklocation of h1b applicants after selection 

data['WORKSITE'].value_counts().head(15).plot(kind='barh')
#employer of h1b applicants after selection 

data['EMPLOYER_NAME'].value_counts().head(15).plot(kind='barh')
#deignation of h1b applicants after selection 

data['JOB_TITLE'].value_counts().head(15).plot(kind='barh')
#top h1b employer in 2016

data_2016= data[data['YEAR']==2016]

data_2016['EMPLOYER_NAME'].value_counts().head(15).plot(kind='barh')
#contract details of 2016 hiring

data_2016['FULL_TIME_POSITION'].value_counts().head(15).plot(kind='barh')
#average wage for certified h1b applicants

data.groupby(['EMPLOYER_NAME']).mean()['PREVAILING_WAGE'].head(20).plot(kind='barh')
#average wage for certified h1b applicants per company

data.groupby(['JOB_TITLE','EMPLOYER_NAME']).mean()['PREVAILING_WAGE'].head(20).plot(kind='barh')
#average wage for certified h1b applicants per year

data.groupby(['JOB_TITLE','YEAR']).mean()['PREVAILING_WAGE'].head(20).plot(kind='barh')
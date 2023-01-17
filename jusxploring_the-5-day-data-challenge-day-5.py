# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as s



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
bites_df = pd.read_csv("../input/Health_AnimalBites.csv")

bites_df.head()
bites_df.shape
#Drop columns..

gender = bites_df.drop(columns = ['vaccination_yrs','vaccination_date','victim_zip','AdvIssuedYNDesc','BreedIDDesc','color','WhereBittenIDDesc','quarantine_date','DispositionIDDesc','head_sent_date','release_date','ResultsIDDesc'])

gender
#gender = [gender[i] for i in range(0,len(gender)) if gender[i] != 'UNKNOWN' or gender[i] != 'nan']

gender = gender.dropna()
#let's just plot it using seaborn's sns.countplot()

data = gender['GenderIDDesc']

sns.countplot(data)
gender_ID = gender['GenderIDDesc']

#REMOVE UNKNOWNS:

gender_ID = [gender_ID[i] for i in list(gender_ID.index) if gender_ID[i] != 'UNKNOWN']
gender_ID_df = pd.DataFrame(gender_ID, columns = ['Gender ID'])

gender_ID_df.head(10)
data = gender_ID_df['Gender ID'].value_counts()

data
s.chisquare(data)
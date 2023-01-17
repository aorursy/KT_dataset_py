# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('../input/exams.csv')

#data.head()

#df.columns
#exsc contains the number of students taking each exam subject

exsc = df.groupby(['Exam Subject'])['Students (Male)','Students (Female)']

ordered_mal = exsc.sum().sort_values(ascending=0,by='Students (Male)')[1:10]

ordered_mal.plot(kind = 'bar')

ordered_fem = exsc.sum().sort_values(ascending=0,by='Students (Female)')[1:10]

ordered_fem.plot(kind = 'bar')



#mal_sub = df['Students (Male)'].groupby(df['Exam Subject'])

#mal = mal_sub.sum().sort_values(ascending=0)

#fem_sub = df['Students (Female)'].groupby(df['Exam Subject'])

#fem = fem_sub.sum().sort_values(ascending=0)

#mal.plot(kind='bar')

#fem.plot(kind='bar')
exsc1 = df.groupby(['Exam Subject'])['Students (Male)','Students (Female)','All Students (2016)'].sum()



#getting the proportion of students by sex per subject

prop_mal = (exsc1['Students (Male)']/exsc1['All Students (2016)']).sort_values(ascending=0)

prop_fem = (exsc1['Students (Female)']/exsc1['All Students (2016)']).sort_values(ascending=0)

#prop_mal.plot(kind='bar')

prop_fem.plot(kind='bar')



#plots the difference between the male and female proportions per subject

#diff = (exsc1['Students (Male)'] - exsc1['Students (Female)'])

#diff = prop_mal - prop_fem

#diff = diff.sort_values(ascending=0)

#diff.plot(kind='bar')
#want to see how the score of each groups compares to the average score

gen = df.groupby(['Exam Subject','Score'])['Students (Male)','Students (Female)'].sum()

gen = gen.filter(like='Average',axis=0)

gen = gen.filter(like='PHYSICS',axis=0)

gen.plot(kind='bar')
#want to see how the score of each groups compares to the average score

eth = df.groupby(['Exam Subject','Score'])['Students (White)','Students (Black)','Students (Hispanic/Latino)','Students (Asian)','Students (American Indian/Alaska Native)','Students (Native Hawaiian/Pacific Islander)'].sum()

eth = eth.filter(like='Average',axis=0)[0:5]

#eth = eth.filter(like='PHYSICS',axis=0)

eth.plot(kind='bar')
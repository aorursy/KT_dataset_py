# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats as st

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
survey = pd.read_csv(r"../input/anonymous-survey-responses.csv")
survey.info()
survey.describe()
survey.head()
survey.columns
sns.countplot(survey['Have you ever taken a course in statistics?'])
# frequencies of attribute in a series

survey['Have you ever taken a course in statistics?'].value_counts()
# run chisquare test for statistic people

st.chisquare(survey['Have you ever taken a course in statistics?'].value_counts())
# frequencies of attribute in a series

survey['Do you have any previous experience with programming?'].value_counts()
sns.countplot(survey['Do you have any previous experience with programming?'])
# run chisquare test for programming people

st.chisquare(survey['Do you have any previous experience with programming?'].value_counts())
contigencyTable = pd.crosstab(survey['Have you ever taken a course in statistics?'],

                              survey['Do you have any previous experience with programming?'])

contigencyTable
chival,pval,dof,exparray = st.chi2_contingency(contigencyTable)

print("Chi Value: " + str(chival))

print("p-Value: " + str(pval))

print("Degree of Freedom Value: " + str(dof))
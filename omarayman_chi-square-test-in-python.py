# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.stats

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

survey = pd.read_csv("../input/anonymous-survey-responses.csv")

survey.head()
scipy.stats.chisquare(survey["Have you ever taken a course in statistics?"].value_counts())
scipy.stats.chisquare(survey["Do you have any previous experience with programming?"].value_counts())
cont = pd.crosstab(survey["Have you ever taken a course in statistics?"],survey["Do you have any previous experience with programming?"])
scipy.stats.chi2_contingency(cont)
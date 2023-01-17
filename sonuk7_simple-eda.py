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

df=pd.read_csv(r'../input/questions.csv')

df.head()
import matplotlib.pyplot as plt

var=df.groupby('level').count()['QCode']

var=var.sort_values(ascending=False)

plt.title('Number of questions by level')

var.plot(kind='bar')
df2=pd.read_csv(r'../input/solutions.csv')

df2.head()
var=df2.groupby('Status').count()

print(var.head())

var.sort_values(inplace=True,by='SolutionID',ascending=False)

plt.clf()

plt.title('Status and Attempt')

var['SolutionID'].head(5).plot.pie(figsize=(5,5),autopct='%.2f')
var=df2[df2['Status']=='accepted'].groupby('UserID').agg({"QCode":pd.Series.nunique})

var=var.sort_values(ascending=False,by='QCode')

var['NumberOfQuestionsSolved']=var['QCode']

del var['QCode']

var.head(5)
var=df2.groupby('QCode').agg({"SolutionID":pd.Series.nunique})

var=var.sort_values(ascending=False,by='SolutionID')

var['NumberOfSolutions']=var['SolutionID']

del var['SolutionID']

var.head()
var=df2[df2['Status']=='accepted'].groupby('Language').count()

print(var.head())

var.sort_values(inplace=True,by='SolutionID',ascending=False)

plt.clf()

plt.title('Most Used Languages for Accepted Solution')

var['SolutionID'].head(5).plot.pie(figsize=(5,5),autopct='%.2f')
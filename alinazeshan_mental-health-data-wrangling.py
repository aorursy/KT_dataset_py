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
#import Libraries

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



#loading dataset

df = pd.read_csv('../input/survey.csv')



#summarize the data

df.describe()



#view first first rows

df.head()
#Interesting Age column

df.Age.unique()
#fixing the Age column first

df = df[(df['Age']> 0) & (df['Age'] < 110)]

df.Age.unique()
# plot histogram of Age

sns.set(color_codes =True)

sns.distplot(df['Age'], kde=False).set_title("Age Distribution")

plt.ylabel('Frequency')
# Gender Count

#df.Gender.value_counts()

#Bad Idea
#Country Distribution

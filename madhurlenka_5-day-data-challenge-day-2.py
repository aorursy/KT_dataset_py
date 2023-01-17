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
#This is 2nd day of the data challenge. I have choosen the data from Titanic: Machine Learning from Disaster competition

#read in CSV's from a file path

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
#import necessary libraries

import seaborn as sns

#lets check out the summary of the data first

train.Age.describe()
train[['Age']].info()
train.Age.fillna(value=train.Age.mean(), inplace=True)
sns.distplot(train.Age, kde=False, bins=10).set_title('Age distribution of Titanic Passengers')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# read the train.ccsv file

train = pd.read_csv("/kaggle/input/titanic/train.csv")

train
# read the gender_submission.csv file

gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

gender_submission
# read the test.csv file

test = pd.read_csv("/kaggle/input/titanic/test.csv")

test
# extract the rows in form of a series where the passanger is a male and has survived

men_l = train.loc[train.Sex=="male"]['Survived']

print(men_l)
# find the percentage of male passangers who survived 

survival_m = men_l.sum()/len(men_l)

print (f'survival rate of Men on the titanic that night = {survival_m * 100} %')
# extract the rows in form of a series where the passanger is a female and has survived

women_l = train.loc[train.Sex=="female"]['Survived']

print(women_l)
# find the percentage of female passangers who survived 

survival_f = women_l.sum()/len(women_l)

print (f'survival rate of Women on the titanic that night = {survival_f * 100} %')
# importing the matplotlib library first

import matplotlib.pyplot as plt
# Bar plots

pd.DataFrame(train.loc[train.Survived==0]['Age']).plot(kind = 'bar', color = 'red', figsize = (20,10))

plt.xticks([])

plt.ylabel('Ages')

plt.title('Age distribution of passangers who did not survive')            # title = title of plot

plt.show()
pd.DataFrame(train.loc[train.Survived==0]['Age']).plot(kind = 'bar', color = 'green', figsize = (20,10))

plt.xticks([])

plt.ylabel('Ages')

plt.title('Age distribution of passangers who survived')            # title = title of plot

plt.show()
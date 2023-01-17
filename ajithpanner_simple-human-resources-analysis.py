# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
hrdata = pd.read_csv("../input/HR_comma_sep.csv")
hrdata.head()
#LETS CHANGE THE SALES COLUMN NAME AS DEPARTMENT

hrdata=hrdata.rename(columns = {'sales':'department'})
hrdata.head()
hrdata.describe()
hrdata.info()
hrdata.corr()
hrdata.groupby('left').mean()['satisfaction_level'].plot(kind='bar',color='g',figsize=(10,5))
last_evaluation_based= hrdata.groupby(['department','left'] ,as_index=False)[['last_evaluation']].mean()

sns.barplot(x='last_evaluation', data=last_evaluation_based, y='department', hue='left')

plt.show()
def mapping(item):

    if item in [0,1,2,3]:

        return "low_time_spend"

    if item in [4,5,6]:

        return "medium_time_spend"

    if item in [7,8,9,10]:

        return "high_time_spend"



hrdata["time_spend"] = hrdata["time_spend_company"].map(mapping)
hrdata["time_spend"].value_counts()
timebased= pd.crosstab([hrdata.time_spend], hrdata.left)

timebased.plot.bar(stacked=True)

plt.show()
hrdata.groupby('left').mean()['average_montly_hours'].plot(kind='bar',color='g',figsize=(10,5))
def mapping(item):

    if item in [2,3]:

        return "low_num_project"

    if item in [4,5]:

        return "medium_num-project"

    if item in [6,7]:

        return "high_num_project"

    

hrdata["num_project"] = hrdata["number_project"].map(mapping)
hrdata["num_project"].value_counts()
projectbased= pd.crosstab([hrdata.num_project], hrdata.left)

projectbased.plot.bar(stacked=True)

plt.show()
workaccident= pd.crosstab([hrdata.promotion_last_5years], hrdata.left)

workaccident.plot.bar(stacked=True)

plt.show()
hrdata["salary"].value_counts()
salarybased= pd.crosstab([hrdata.salary], hrdata.left)

salarybased.plot.bar(stacked=True)

plt.show()
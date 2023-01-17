# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/StudentsPerformance.csv')
data.info()
data.head()
#Correlation Table

data.corr()
#Correlation Map

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(), annot=True, linewidth=.5, fmt='.2f', ax=ax)

plt.show()
NewColumnName=['gender','Ethnicity','ParentalLevelofEducation','lunch','TestPreparationCourse','MathScore','ReadingScore','WritingScore']

data.columns=NewColumnName
data.columns
#Scatter Plot

none=data[data.TestPreparationCourse=="none"]

completed=data[data.TestPreparationCourse=="completed"]

plt.scatter(none.ReadingScore,none.WritingScore,color='red',label='None',alpha=0.4)

plt.scatter(completed.ReadingScore,completed.WritingScore,color='green',label='Completed',alpha=0.7)

plt.legend()

plt.xlabel("Reading Score")

plt.ylabel("Writing Score")

plt.title("Test Preparation Course Plot")

plt.show()
# Histogram 

data.MathScore.plot(kind='hist',bins=50,figsize=(12,12))

plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas  as pand

import matplotlib.pyplot as plt

import seaborn as seab



import numpy as np

import pandas as pd

import warnings

warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



data=pand.read_csv("../input/income_evaluation.csv")



#we don't require these columns in our dataset(useless)

data=data.drop([' fnlwgt',' native-country',' education-num'],axis='columns')

dummall=pd.get_dummies(data)

t=[dummall[' income_ <=50K'],dummall[' income_ >50K']]

Y=pd.DataFrame(t)

Y= Y.transpose()

X=dummall.drop([' income_ <=50K',' income_ >50K'],axis='columns')





#preliminary analysis

#Analyzing our dataset for reaching our goal i.e findings











pd.crosstab(data[' occupation'],data[' income']).plot(kind="bar",figsize=(22,6),color=['#4e77b7','#ef81ef' ])

plt.title('Occupation vs Income')

plt.xlabel('Occupation')

plt.xticks(rotation=0)

plt.ylabel('Frequency')

plt.show()



#From  the  figure we get to know that exec¬managerial and prof¬specialty 

#stand out as having very high percentages of individuals making over $50,000. In addition, the percentages for 

#Farming¬fishing, Other¬service and Handlers¬cleaners are significantly lower than the rest of the distribution.

















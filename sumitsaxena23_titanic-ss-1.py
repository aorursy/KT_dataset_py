# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head(20)
# There are some missing values in class. Replace them with a 4th class, distinct from 1st,2nd and 3rd
train['Pclass'].fillna(4)
train['Pclass'] = train['Pclass'].astype(str)
# Overall, the plan is to create a categorical variable which classifies each passenger on Sex+Class+age
# To begin with create a new column with merged categorical varibles from Sex and Clss
train['Sex_Class']= train['Sex']+"_"+train['Pclass']
# As age is a continous variable, convert it into a categorical one
train['age_cuts'] = pd.cut(x=train.Age,bins=[0,18,75,100], labels=["Young","Adult","Old"] )
# Checking the survival rates of with age cut
train['age_cuts'] = train['age_cuts'].astype(object)
train.pivot(columns='age_cuts',values='Survived').mean()*100
# creating a new column that incorporates Sex+Class+Age classification
train['Sex_Class_Age']= train['Sex_Class']+"_"+train['age_cuts']
# Generating survival probability for each Sex+Class+Age groups
train_table = pd.DataFrame(train.pivot(columns='Sex_Class_Age',values= 'Survived').mean()*100)
# preparing a new table with survival probabilites. This table would be merged into the original DF 
train_table.reset_index(inplace=True)
train_table.columns = ['Sex_Class_Age','Predict_Score']
train_table.head()
# Merging the dataframes to have survival probabilty for each passenger
train =train.merge(train_table,how='left', on='Sex_Class_Age') 
# If the survival probability is more than 50%, the prediction is survival, otherwise non-survival
train['Survival_Predict']=0
train['Survival_Predict'].loc[train.Predict_Score>50]=1

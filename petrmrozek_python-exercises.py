import numpy as np 

import pandas as pd 

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#chapter2_exercise8a

import pandas as pd

College = pd.read_csv("../input/college/College.csv")

#chapter2_exercise8b

numbered_college = College.set_index(['Unnamed: 0'], append=True, verify_integrity=True)

numbered_college.rename_axis([None, 'College'], inplace=True)

numbered_college.head()
#chapter2_exercise8cii

numbered_college.describe()
#chapter2_exercise8cii

sns.pairplot(numbered_college.iloc[:, 1:11]);
#chapter2_exercise8ciii

sns.boxplot(x=numbered_college['Private'], y=numbered_college['Outstate']);
#chapter2_exercise8civ

numbered_college['Elite'] = numbered_college['Top10perc'] > 50



numbered_college['Elite'].sum()

sns.boxplot(x=numbered_college['Elite'], y=numbered_college['Outstate']);
#chapter2_exercise8cv

numbered_college.hist(['Room.Board','Books', 'Personal', 'Expend' ],bins=100, alpha=0.5, figsize=(8,5));
#chapter2_exercise8cvi

numbered_college.boxplot(column='Expend',by='Elite', figsize=(5,5));

#the comparison of the expenditure of elite school
#chapter3_exercise12a

#if sum of i=1^(n)x_i is equal to sum of i=1^(n)y_i
import pandas as pd

import numpy as np

import statsmodels.api as sm

auto = pd.read_csv("../input/autopy/Auto.csv")

auto.head()



#chapter3_exercise12b



np.random.seed(101)

x = np.random.randn(100)

y = x + np.random.randn(100)





model1 = sm.OLS(y,x)

estimate1 = model1.fit()

print("The coeffecient is:", estimate1.params)



model1 = sm.OLS(x,y)

estimate1 = model1.fit()

print("The coeffecient is:", estimate1.params)
#chapter3_exercise12c

x = np.random.randn(100)

y = np.random.choice(x,100,replace=False)



model2 = sm.OLS(y,x)

estimate2 = model2.fit()

print("The coeffecient is:", estimate2.params)



model2 = sm.OLS(x,y)

estimate2 = model2.fit()

print("The coeffecient is:", estimate2.params)
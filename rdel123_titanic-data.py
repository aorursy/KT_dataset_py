# include modules 

# these are the most common ones we have used in the Big Data Class I am taking 

import math as m

import numpy as np

import scipy as sp

import pandas as pd

import matplotlib as plt

import matplotlib.pyplot as plt

import seaborn as sns  #IMPORTANT - UPGRADE SEABORN TO VERSION 0.9.0 IN ANACONDA ENV

import statsmodels.api as sm

import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier



# loading in packages



import numpy as np 

import pandas as pd  #(e.g. pd.read_csv) going to read in the csv so I can actually use it 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
train_data.info() 
train_data.describe()
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()

# want to see the head so I can see the data
print(train_data.columns.values)




test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()



women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)



# shows me what percentage of women survived which is helpful but I want to add more.
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)

import matplotlib.pyplot as plt



sex_pivot = train_data.pivot_table(index="Sex",values="Survived")

sex_pivot.plot.bar()

plt.show()
class_pivot = train_data.pivot_table(index="Pclass",values="Survived")

class_pivot.plot.bar()

plt.show()




from sklearn.ensemble import RandomForestClassifier



y = train_data["Survived"]



features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])

X_test = pd.get_dummies(test_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")



g = sns.FacetGrid(train_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)





# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();



X_train = train_data.drop("Survived", axis=1)

Y_train = train_data["Survived"]

X_test  = test_data.drop("PassengerId", axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape

from sklearn.ensemble import RandomForestClassifier 



# y is the labels 

y = train_data ["Survived"]



# will use these features to train the data



features = ["Sex", "Pclass"]



X = pd.get_dummies(train_data[features])



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)



model.fit(X, y)



predictions = model.predict(X_test)



output_andomF = pd.dataFrame({"cabin": test_data.cabin, "Survived": predictions})



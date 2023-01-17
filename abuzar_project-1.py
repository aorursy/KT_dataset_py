# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





#visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.







#import train and test CSV files

train = pd.read_csv("../input/Data.csv")

test = pd.read_csv("../input/Data.csv")



#take a look at the training data

train.describe(include="all")
#get a list of the features within the dataset

print(train.columns)



#see a sample of the dataset to get an idea of the variables

train.sample(5)
#see a summary of the training dataset

train.describe(include = "all")
#check for any other unusable values

print(pd.isnull(train).sum())
#draw a bar plot of participation by gender

sns.barplot(x="Gender", y="Participation", data=train)



#print percentages of females vs. males that attend program

print("Percentage of females who attend:", train["Participation"][train["Gender"] == 'F'].value_counts(normalize = True)[1]*100)



print("Percentage of males who attend:", train["Participation"][train["Gender"] == 'M'].value_counts(normalize = True)[1]*100)

#draw a bar plot for Mobility vs. Participation

sns.barplot(x="Mobility", y="Participation", data=train)

plt.show()
from sklearn.model_selection import train_test_split



predictors = train.drop(['Reason'], axis=1)

target = test["Participation"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
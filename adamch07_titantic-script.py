# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#Plotting

import seaborn as sns



#Linear Models

import statsmodels.formula.api as smf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

gs_df = pd.read_csv('../input/gender_submission.csv')
train_df.head()
test_df.head()
train_copy = train_df.copy()



#Replace Males with 1 and Females with 0

train_copy['Sex'].replace('male', '1', inplace=True)

train_copy['Sex'].replace('female', '0', inplace=True)



sns.pairplot(train_copy, x_vars=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'], y_vars=['Survived'])
multiModel = smf.ols('Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare', data = train_copy).fit()

print(multiModel.summary())
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
import pandas as pd

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
#Survived

f, ax =plt.subplots(2,2, figsize=(15,10))



sns.set(style='whitegrid')

sns.countplot(x = train['Survived'], palette='Reds', ax=ax[0,0])

ax[0,0].set_title('Count of Survived')

sns.swarmplot(x='Survived', y='Fare', data=train, ax=ax[0,1])

ax[0,1].set_title('Survived vs Fare')

sns.countplot(x = train['Survived'], hue=train['Sex'], palette='Reds', ax=ax[1,0])

ax[1,0].set_title('Survived vs Sex')

sns.countplot(x = train['Survived'], hue=train['Pclass'], palette='Reds', ax=ax[1,1])

ax[1,1].set_title('Survived vs Pclass')



plt.show()
#Pclass

f, ax =plt.subplots(2,2, figsize=(15,10))



sns.countplot(x = 'Pclass', data=train, palette='Blues', ax=ax[0,0])

ax[0,0].set_title('Count of Pclass')

sns.swarmplot(x= 'Pclass', y='Fare', data=train, ax=ax[0,1])

ax[0,1].set_title('SWARM PLOT')

sns.boxplot(x= 'Pclass', y='Fare', data=train, ax=ax[1,0])

ax[1,0].set_title('BOX PLOT')

sns.violinplot(x= 'Pclass', y='Fare',hue='Sex', data=train,split=True, ax=ax[1,1])

ax[1,1].set_title('VIOLIN PLOT')
#Sex

f, ax =plt.subplots(2,2, figsize=(15,10))



sns.countplot(x = 'Sex', data=train, palette='Blues', ax=ax[0,0])

ax[0,0].set_title('Count of Sex')

sns.swarmplot(x= 'Sex', y='Age', data=train, ax=ax[0,1])

ax[0,1].set_title('SWARM PLOT')

sns.pointplot(x= 'Sex', y='Survived',hue='Pclass', data=train, ax=ax[1,0])

ax[1,0].set_title('POINT PLOT')

sns.violinplot(x= 'Sex', y='Age',hue='Pclass', data=train, ax=ax[1,1])

ax[1,1].set_title('VIOLIN PLOT')
#Fare Feature

sns.distplot(train['Fare'])
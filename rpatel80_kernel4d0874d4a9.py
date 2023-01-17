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

        

import matplotlib.pyplot as plt         

from sklearn.linear_model import LogisticRegression

%matplotlib inline

# Any results you write to the current directory are saved as output.
#loading in data 

#trd= pd.read_csv('/kaggle/input/titanic/train.csv', index_col=0)

#test_data = pd.read_csv('/kaggle/input/titanic/test.csv',index_col=0)

trd= pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')

trd.head()
#Exploratory visualizations age

trd.Age.plot(kind='hist')

plt.scatter(trd.Pclass, trd.Fare)

plt.show()
#Exploratory visualizations Pclass vs. Fare

plt.scatter(trd.Pclass, trd.Fare)

plt.show()
#Exploratory visualizations crosstab survive and pclass

pd.crosstab(trd.Survived, trd.Pclass)
trd[(trd.Pclass == 1) & (trd.Age <10)]

x = pd.Series(np.zeros(len(test_data),dtype='int64'))
temp = pd.DataFrame({'PassengerId':test_data.PassengerId, 'Survived':x, 'Pclass':test_data.Pclass, 'Age':test_data.Age})

temp.head()
temp =[(temp.Age < 10) & (temp.Pclass == 1)]
temp.loc[(temp.Age<10) & (temp.Pclass == 1),'Survived']= 1
temp[(temp.Age<10) & (temp.Pclass == 1)]
sub = pd.DataFrame({'PassengerId':temp.PassengerId, 'Survived':temp.Survived})

sub.head()
sub[sub.Survived == 1]
sub.to_csv('sub.csv')
final = pd.DataFrame({'PassengerId':sub.PassengerId, 'Survived':sub.Survived}).set_index('PassengerId').to_csv('sub.csv')
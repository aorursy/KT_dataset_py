# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import preprocessing
%matplotlib notebook
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

plt.rcParams [ 'axes.labelsize' ] = 18
plt.rcParams [ 'lines.linewidth' ] = 4
plt.rcParams [ 'figure.figsize' ] = ( 8 , 5)
plt.rcParams [ 'font.size' ] = 9
path_test = r'../input/titanic/test.csv'
path_train = r'../input/titanic/train.csv'

df_test = pd.read_csv(path_test)
df_train = pd.read_csv(path_train)
df_test.head()
df_train.head()
%matplotlib inline
fig = plt.figure(figsize = (14, 8))
ax1 = fig.add_subplot(221)
ax1.set_title('Survived')
df_train['Survived'].value_counts().plot.pie( autopct = "%.2f%%", colors = [ 'cyan', 'red' ] )
ax2 = fig.add_subplot(222)
ax2.set_title('Percetage of males vs females')
df_train['Sex'].value_counts().plot.pie( autopct = "%.2f%%", colors = [ 'cyan', 'pink' ] )

ax3 = fig.add_subplot(223)
ax3.set_title('Survived males')
df_train['Survived'][ df_train['Sex'] == 'male' ].value_counts().plot.pie( autopct = "%.2f%%", colors = [ 'cyan', 'red' ] )


ax4 = fig.add_subplot(224)
ax4.set_title('Survived females')
df_train['Survived'][ df_train['Sex'] == 'female' ].value_counts().plot.pie( autopct = "%.2f%%", colors = [ 'cyan', 'red' ] )
plt.show()
df_train.describe()
le = preprocessing.LabelEncoder()
le.fit(df_train['Sex'] )
df_train['Sex'] = le.transform( df_train['Sex'] ) # female = 0, male = 1
df_train.head()
u = pd.get_dummies(df_train['Embarked']) # C = Cherbourg, Q = Queenstown (Ireland), S = Southampton
u.rename( columns = {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}, inplace = True )

u.head()
#pd.merge([df_train, u], axis = 0).head()
col = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
stat = df_train[ col ].corr()
stat
correlations = abs( df_train[col].corr()['Survived']).sort_values()
print('Most Correlated characters:\n', correlations)
correlations = df_train[col].corr()['Survived'].sort_values()
print('Most Correlated characters:\n', correlations)
stat

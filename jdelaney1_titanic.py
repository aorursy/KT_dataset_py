# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
fname = '../input/train.csv'

data = pd.read_csv(fname)
data.head()

data.count()
data['Age'].min(), data['Age'].max()
data['Survived'].value_counts()
%matplotlib inline



alpha_color = 0.5



data['Survived'].value_counts().plot(kind='bar' )



data.plot(kind='scatter', x='Survived', y='Age')
data[data['Survived'] == 1]['Age'].value_counts().sort_index().plot(kind='bar')
bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]



data['AgeBin'] = pd.cut(data['Age'], bins)



#How did age factor in to survival?
data[data['Survived'] == 1]['AgeBin'].value_counts().sort_index().plot(kind='bar')
data[data['Survived'] == 0]['AgeBin'].value_counts().sort_index().plot(kind='bar')
data['AgeBin'].value_counts().sort_index().plot(kind='bar')
#Age distribution is about the same for passangers who survived, died, and overall
#What about class?
data[data['Pclass'] == 1]['Survived'].value_counts().plot(kind='bar')
data[data['Pclass'] == 3]['Survived'].value_counts().plot(kind='bar')
# most of the third class did not survive
#what about gender?
data[data['Sex'] == 'female']['Survived'].value_counts().plot(kind='bar')
data[data['Sex'] == 'male']['Survived'].value_counts().plot(kind='bar')
#Most men did not survive, most women did



#Now compare sex and class. How many 3rd class men survived?
data[(data['Sex'] == 'male') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')
data[(data['Sex'] == 'male') & (data['Pclass'] == 1)]['Survived'].value_counts().plot(kind='bar')
#males in 1st class did better, but still most did not survive



# What about females?
data[(data['Sex'] == 'female') & (data['Pclass'] == 1)]['Survived'].value_counts().plot(kind='bar')
data[(data['Sex'] == 'female') & (data['Pclass'] == 3)]['Survived'].value_counts().plot(kind='bar')
#Nearly all first class women survived and about hallf of the third class women survived
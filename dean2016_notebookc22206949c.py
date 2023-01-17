# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn.linear_model as lm

import matplotlib.pyplot as plt 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print ("Dimension of train data {}".format(train.shape))

print ("Dimension of test data {}".format(test.shape))
print ('Basic data description')

train.describe()
train.tail()
test.head()
plt.rc('font', size=13)

fig = plt.figure(figsize=(18, 8))

alpha = 0.6



ax1 = plt.subplot2grid((2,3), (0,0))

train.Age.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)

test.Age.value_counts().plot(kind='kde', label='test', alpha=alpha)

ax1.set_xlabel('Age')

ax1.set_title("What's the distribution of age?" )

plt.legend(loc='best')



ax2 = plt.subplot2grid((2,3), (0,1))

train.Pclass.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=alpha)

test.Pclass.value_counts().plot(kind='barh', label='test', alpha=alpha)

ax2.set_ylabel('Pclass')

ax2.set_xlabel('Frequency')

ax2.set_title("What's the distribution of Pclass?" )

plt.legend(loc='best')



ax3 = plt.subplot2grid((2,3), (0,2))

train.Sex.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=alpha)

test.Sex.value_counts().plot(kind='barh', label='test', alpha=alpha)

ax3.set_ylabel('Sex')

ax3.set_xlabel('Frequency')

ax3.set_title("What's the distribution of Sex?" )

plt.legend(loc='best')



ax4 = plt.subplot2grid((2,3), (1,0), colspan=2)

train.Fare.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)

test.Fare.value_counts().plot(kind='kde', label='test', alpha=alpha)

ax4.set_xlabel('Fare')

ax4.set_title("What's the distribution of Fare?" )

plt.legend(loc='best')



ax5 = plt.subplot2grid((2,3), (1,2))

train.Embarked.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=alpha)

test.Embarked.value_counts().plot(kind='barh', label='test', alpha=alpha)

ax5.set_ylabel('Embarked')

ax5.set_xlabel('Frequency')

ax5.set_title("What's the distribution of Embarked?" )

plt.legend(loc='best')

plt.tight_layout()

print(train.Survived.value_counts())
fig = plt.figure(figsize=(8, 4))



train[train.Survived==0].Age.value_counts().plot(kind='density', color='#FA2379', label='Not Survived', alpha=alpha)

train[train.Survived==1].Age.value_counts().plot(kind='density', label='Survived', alpha=alpha)

plt.xlabel('Age')

plt.title("What's the distribution of Age?" )

plt.legend(loc='best')

plt.grid()

df_male = train[train.Sex=='male'].Survived.value_counts().sort_index()

df_female = train[train.Sex=='female'].Survived.value_counts().sort_index()



fig = plt.figure(figsize=(18, 6))

ax1 = plt.subplot2grid((1,2), (0,0))

df_female.plot(kind='barh', color='#FA2379', label='Female', alpha=alpha)

df_male.plot(kind='barh', label='Male', alpha=alpha-0.1)

ax1.set_xlabel('Frequrncy')

ax1.set_yticklabels(['Died', 'Survived'])

ax1.set_title("Who will survived with respect to sex?" )

plt.legend(loc='best')

plt.grid()



ax2 = plt.subplot2grid((1,2), (0,1))

(df_female/train[train.Sex=='female'].shape[0]).plot(kind='barh', color='#FA2379', label='Female', alpha=alpha)

(df_male/train[train.Sex=='male'].shape[0]).plot(kind='barh', label='Male', alpha=alpha-0.1)

ax2.set_xlabel('Rate')

ax2.set_yticklabels(['Died', 'Survived'])

ax2.set_title("What's the survived rate with respect to sex?" )

plt.legend(loc='best')

plt.grid()
plt.rc('font', size = 5)

df_male = train[train.Sex=='male']

df_female = train[train.Sex=='female']

fig = plt.figure(figsize = (18, 6))



ax1 = plt.subplot2grid((1,4), (0,0))

df_female[df_female.Pclass<3].Survived.value_counts().sort_index().plot(kind='bar', color='#FA2379', alpha=alpha)

ax1.set_ylabel('Number')

ax1.set_ylim((0,350))

ax1.set_xticklabels(['Died', 'Survived'])

ax1.set_title('Number of survived or died for female in high-class')

plt.grid()



ax2 = plt.subplot2grid((1,4), (0,1))

df_female[df_female.Pclass==3].Survived.value_counts().sort_index().plot(kind='bar', color='r',alpha=alpha)

ax2.set_ylabel('Number')

ax2.set_ylim((0,350))

ax2.set_xticklabels(['Died', 'Survived'])

ax2.set_title('Number of survived or died for female in lower-class')

plt.grid()



ax3 = plt.subplot2grid((1,4), (0,2))

df_male[df_male.Pclass<3].Survived.value_counts().sort_index().plot(kind = 'bar', color = 'b', alpha = alpha)

ax3.set_ylabel('Number')

ax3.set_ylim((0, 350))

ax3.set_xticklabels(['Died', 'Survived'])

ax3.set_title('Number of survived or died for male in high-class')

plt.grid()



ax4 = plt.subplot2grid((1,4), (0,3))

df_male[df_male.Pclass==3].Survived.value_counts().sort_index().plot(kind = 'bar', color = 'b', alpha = alpha)

ax4.set_ylabel('Number')

ax4.set_ylim((0, 350))

ax4.set_xticklabels(['Died', 'Survived'])

ax4.set_title('Number of survived or died for male in high-class')

plt.grid()

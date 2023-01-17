# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Loading data, dividing, modeling and EDA below

from sklearn.model_selection import train_test_split

# numpy, matplotlib, seaborn

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
test_df = pd.read_csv("../input/test.csv")

train_df = pd.read_csv("../input/train.csv")
train_df.info()
#we have to first convert the categorical features to numerical form

#step ! : finding the missing values and fill them

total = train_df.isnull().sum().sort_values(ascending=False)

percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100

percent_2 = (round(percent_1, 1)).sort_values(ascending=False)

#print([total,percent_2]) # as two different elemets of the list

missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])

print(missing_data)
#drop the columns having the highest number of missing values and the ones that do not affect the data too much

train_df = train_df.drop(['PassengerId','Name','Ticket'], axis=1)

X_test  = test_df.drop(["PassengerId",'Name','Ticket'], axis=1).copy()

test_df = test_df.drop(['PassengerId','Name','Ticket'], axis=1)
train_df.head()
#converting variables like 'C133' to numeric form

import re

deck = {"O": 0,"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8 }

data = [train_df, test_df] # two distinct datasets



for i in data:

    i['Cabin'] = i['Cabin'].fillna('O')  #handle the missing values first

    regex_num = re.compile('[A-Z]+')

    i['Deck']= i['Cabin'].map(lambda x: regex_num.search(x).group(0))

    i['Deck']=i['Deck'].map(deck)

    i['Deck']=i['Deck'].fillna(0)

    i['Deck']=i['Deck'].astype(int)

#train_df['Deck'].astype(int)

    

#print(train_df.index[train_df['Deck'].apply(np.isnan)])

#print(train_df['Deck'].iloc[399])



#find pclass passengers in each deck

train_df1=train_df[train_df.Survived > 0]   # only select the values where survived=1

train_df1.pivot_table(values='Survived',index=['Deck' , 'Pclass'],aggfunc='count')  # this can be done using pivot _table

#train_df.info()
#dropping the cabin column

train_df = train_df.drop(['Cabin'],axis=1)

test_df = test_df.drop(['Cabin'],axis=1)



#a plot of pclass and survived

sns.barplot(x='Pclass', y='Survived', data=train_df)

#what was the age of the people split by the different classes in ship who survived the Titanic disaster

bins = np.arange(0, 65, 5)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid=grid.map(plt.hist , 'Age' , bins= bins , color='m' )
#train_df

data = [train_df, test_df]

missing_per=(train_df['Age'].isnull().sum()/train_df['Age'].isnull().count())*100

print("Age has " , missing_per , "% values")

sns.violinplot(x='Age' , data =train_df) 
# PLotting the age of the people who survived the titanic 



fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(10, 6))

women = train_df[train_df['Sex']=='female']  # all values in dataset with the sex = 'female'

men = train_df[train_df['Sex']=='male']

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[0,0], kde =False)

ax.set_title('Female survived')

ax.legend()



ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[0,1], kde =False)

ax.set_title('Female not survived')

ax.legend()



ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = 'survived', ax = axes[1,0], kde = False)

ax.set_title('Male survived')

ax.legend()





ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = 'not_survived', ax = axes[1,1], kde = False)

ax.set_title('Male not survived')

ax.legend()



# plotting the frequency count of age

data=train_df.groupby(['Age'])

#new=pd.DataFrame([,)



# arranging the age in the form of age groups



#handling missing age values

# handling missing values of age column

data = [train_df, test_df]



for dataset in data:

    mean = train_df["Age"].mean()

    std = test_df["Age"].std()

    is_null = dataset["Age"].isnull().sum()

    # compute random numbers between the mean, std and is_null

    rand_age = np.random.randint(mean - std, mean + std, size = is_null) # generates a list of random numbers

    # fill NaN values in Age column with random values generated

    age_slice = dataset["Age"].copy()

    age_slice[np.isnan(age_slice)] = rand_age

    dataset["Age"] = age_slice

    dataset["Age"] = train_df["Age"].astype(int)
# plotting the frequency count of age

import copy

import matplotlib.pyplot as plt



data = [train_df,test_df]

for dataset in data:

    #dataset['Age'] = dataset['Age'].fillna(-1)

    dataset['Age'] = dataset['Age'].astype(int)

    dataset.loc[ dataset['Age'] <= 11 , 'Age'] = 0

    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

    

# now we can get an idea of which age group survived in titanic disaster



dataset=data[0]

ax=pd.crosstab(dataset.Age,dataset.Survived).plot(kind='bar')

ax.set_xticklabels(['<11' , '11 -18' , '18 - 22' , '22 - 27' , '27 - 33' , '33 - 40' , ' 40 - 66' , '> 66 '])

# the same procedure can be repeated for fare column 

data = [train_df, test_df]



for dataset in data:

    dataset['Fare'] = dataset['Fare'].fillna(0)

    dataset['Fare'] = dataset['Fare'].astype(int)





#data = [train_df, test_df]





for dataset in data:

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4

    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5

    dataset['Fare'] = dataset['Fare'].astype(int)
genders = {"male": 0, "female": 1}

data = [train_df, test_df]



for dataset in data:

    dataset['Sex'] = dataset['Sex'].map(genders)

    dataset['Sex'] = dataset['Sex'].astype(int)



train_df.head()
# for machine learning models we will require that the categorical variables be converted to numerical variables



# handling missing values for embarked

common_value = 'S'

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(common_value)



ports = {"S": 0, "C": 1, "Q": 2}

#data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].map(ports)

    dataset['Embarked'] = dataset['Embarked'].astype(int)

train_df.head()
train_df.info()
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]



from sklearn.ensemble import RandomForestClassifier





# Random Forest

random_forest = RandomForestClassifier(n_estimators=100, n_jobs= -1, random_state=2)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(test_df)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")

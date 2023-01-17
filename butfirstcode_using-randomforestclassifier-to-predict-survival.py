import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
display(data_train.sample(3))

display(data_test.sample(3))
display(data_train.head())

display(data_test.head())

display(data_test.tail())
data_train.describe() 
data_test.describe()
print(str(data_train.shape)+ ' -> data_train')

print(str(data_test.shape)+ ' -> data_test')
data_train.info()
data_train = data_train.drop(columns=['Name', 'Ticket', 'Fare', 'Cabin']) # dropping columns which are unnecessary for analysis

data_test = data_test.drop(columns=['Name', 'Ticket', 'Fare', 'Cabin']) # doing same for data_test to maintain similar structure of dataframes for both train and test sets

display(data_train.Age.value_counts(dropna=False).sort_index())

display(data_test.Age.value_counts(dropna=False).sort_index())
data_train.Age = data_train.Age.fillna(data_train.Age.mean()) #filling all nulls in 'Age' column with the mean age

data_test.Age = data_test.Age.fillna(data_test.Age.mean()) #filling all nulls in 'Age' column with the mean age



display(data_train.Embarked.value_counts(dropna=False))

display(data_test.Embarked.value_counts(dropna=False))

data_train.Embarked = data_train.Embarked.fillna('S') #filling all nulls in 'Embarked' column with 'S'

b = data_train.pop('Survived') # from data_train, pop the 'Survived' column 

data_train = pd.concat([data_train, b], axis=1) # and add it to the end of data_train

display(data_train.head())

display ( data_train.Age.value_counts(dropna=False).sort_index() )

display ( data_test.Age.value_counts(dropna=False).sort_index() )
display ( data_train.Sex.value_counts(dropna=False).sort_index() )

display( data_test.Sex.value_counts(dropna=False).sort_index() )
display ( (data_train.Pclass.value_counts(dropna=False).sort_index()) )

display ( (data_test.Pclass.value_counts(dropna=False).sort_index()) )
display ( data_train.Survived.value_counts(dropna=False).sort_index() )

display ( data_train.SibSp.value_counts(dropna=False).sort_index() )

display ( data_test.SibSp.value_counts(dropna=False).sort_index() )
display ( data_train.Parch.value_counts(dropna=False).sort_index() )

display ( data_test.Parch.value_counts(dropna=False).sort_index() )
display ( data_train.Embarked.value_counts(dropna=False).sort_index() )

display ( data_test.Embarked.value_counts(dropna=False).sort_index() )
import seaborn as sns

import matplotlib.pyplot as plt



# notes : 

# sns.set()

# f, ax = plt.subplots(figsize=(19, 19))

# sns.heatmap(data_train, annot=True, linewidths=.5, ax=ax)

# -- above not working -- 





print( ' data_train ')



sns.set(style="dark")

# Compute the correlation matrix

corr = data_train.corr()

display(corr)

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(2, 900, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,

            square=True, linewidths=.05, linecolor='grey') 











print( ' data_test ')



sns.set(style="dark")

# Compute the correlation matrix

corr = data_test.corr()

display(corr)

# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap

cmap = sns.diverging_palette(2, 900, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, cmap=cmap, vmax=1, vmin=-1, center=0,

            square=True, linewidths=.05, linecolor='grey') 







plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('TRAIN')

my_palette = {1:'g', 2:'c', 3:'y'}

sns.boxplot(data_train['Pclass'], data_train['Age'], palette=my_palette, saturation= 40) 

plt.xticks = [1,2,3]



plt.subplot(1,2,2)

plt.title('TEST')

my_palette = {1:'g', 2:'c', 3:'y'}

sns.boxplot(data_test['Pclass'], data_test['Age'], palette=my_palette, saturation= 40) 

plt.xticks = [1,2,3]

plt.figure(figsize=(30,20))

plt.subplot(1,2,1)

plt.title('TRAIN')

plt.scatter( data_train['Pclass'], data_train['Age'], c='red', marker='d', s= 6.0)

plt.xticks = ([1,2,3])



plt.subplot(1,2,2)

plt.title('TEST')

plt.scatter( data_test['Pclass'], data_test['Age'], c='red', marker='d', s= 6.0)

plt.xticks = ([1,2,3])
display ( sns.countplot(x=data_train['Survived'], hue=data_train['Sex'], data=data_train, palette='Spectral', saturation=10) )

plt.figure(figsize=(30,20))

plt.subplot(2,6,1)

plt.title('EMBARKED VS. SURVIVAL')

sns.countplot(x=data_train['Embarked'], hue=data_train['Survived'], palette='winter')

plt.ylim(0,700)

plt.legend()

plt.subplot(2,6,2)

plt.title('EMBARKED VS. SEX')

sns.countplot(x=data_train['Embarked'], hue=data_train['Sex'], palette='spring')

plt.ylim(0,700)



# plt.figure(figsize=(5,5))

# sns.countplot(x=data_train['Survived'], hue=data_train['Embarked'], palette= 'summer', alpha=0.3)

# plt.ylim(0,600) -- this code will put two graphs in one -- 



plt.subplot(2,6,3)

plt.title('SURVIVAL VS. SEX')

sns.countplot(x=data_train['Survived'], hue=data_train['Sex'], palette= 'autumn')

plt.ylim(0,700)



plt.subplot(2,6,4)

plt.title('Distribution of Gender')

sns.countplot(x=data_train['Sex'], hue=data_train['Sex'], palette= 'summer')

plt.ylim(0,700)



plt.subplot(2,6,5)

plt.title('Distribution of Embarked Location')

sns.countplot(x=data_train['Embarked'], hue=data_train['Embarked'], palette= 'summer')

plt.ylim(0,700)





plt.subplot(2,6,6)

plt.title('Distribution of Survival')

sns.countplot(x=data_train['Survived'], hue=data_train['Survived'], palette= 'summer')

plt.ylim(0,700)
plt.figure(figsize=(30,20))



plt.subplot(2,6,2)

plt.title('EMBARKED VS. SEX')

sns.countplot(x=data_test['Embarked'], hue=data_test['Sex'], palette='spring')

plt.ylim(0,700)



plt.subplot(2,6,4)

plt.title('Distribution of Gender')

sns.countplot(x=data_test['Sex'], hue=data_test['Sex'], palette= 'summer')

plt.ylim(0,700)



plt.subplot(2,6,5)

plt.title('Distribution of Embarked Location')

sns.countplot(x=data_test['Embarked'], hue=data_test['Embarked'], palette= 'summer')

plt.ylim(0,700)

plt.figure(figsize=(20,6))

plt.subplot(1,3,1)

plt.title('data_train : CLASS VS. SURVIVAL')

sns.countplot(x=data_train['Pclass'], hue=data_train['Survived'], palette='cool')

plt.ylim(0,700)

plt.legend(['Died','Survived'])



plt.subplot(1,3,2)

plt.title('data_train : CLASS VS. GENDER')

sns.countplot(x=data_train['Pclass'], hue=data_train['Sex'], palette='magma')

plt.ylim(0,700)

plt.legend(['Male', 'Female'])



plt.subplot(1,3,3)

plt.title('data_train : GENDER VS. SURVIVAL')

sns.countplot(x=data_train['Sex'], hue=data_train['Survived'], palette='prism')

plt.ylim(0,700)

plt.legend(['Died', 'Survived'])





plt.figure(figsize=(10,10))

plt.subplot(1,1,1)

plt.title('data_test : CLASS VS. GENDER')

sns.countplot(x=data_test['Pclass'], hue=data_test['Sex'], palette='magma')

plt.ylim(0,300)

plt.legend(['Male', 'Female'])
data_train[(data_train.Embarked=='S')].groupby(['Pclass', 'Sex']).size() #.plot(kind='bar', cmap='summer')
data_test[(data_test.Embarked=='S') ].groupby(['Pclass', 'Sex']).size()#.plot(kind='bar', cmap='summer')
data_train.shape
data_train.head()
data_test.shape
data_test.head()
data_train = pd.get_dummies(data_train, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])

data_train.info()
data_test = pd.get_dummies(data_test, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked'])

data_test.info()
data_train['Age'] = (data_train.Age//10*10)
data_test['Age'] = (data_test.Age//10*10)
data_train = pd.get_dummies(data_train, columns=['Age'])
data_test = pd.get_dummies(data_test, columns=['Age'])
b = data_train.pop('Survived')

data_train = pd.concat([data_train, b], axis=1)

data_train.head()
X = data_train.drop(columns = ['Survived', 'PassengerId'], axis=1)
X
y = data_train.Survived
y
# X   #418 rows

# y   #891 rows



data_train.info()
y.head()
plt.figure(figsize=(8,8))

plt.title('data_train : CLASS 1 VS. SURVIVAL')

sns.countplot(x=data_train['Pclass_1'], hue=data_train['Survived'], palette='magma')

plt.ylim(0,700)

plt.legend(['Did not survive', 'Survived'])
plt.figure(figsize=(8,8))

plt.title('data_train : CLASS 2 VS. SURVIVAL')

sns.countplot(x=data_train['Pclass_2'], hue=data_train['Survived'], palette='magma')

plt.ylim(0,700)

plt.legend(['Did not survive', 'Survived'])
plt.figure(figsize=(8,8))

plt.title('data_train : CLASS 3 VS. SURVIVAL')

sns.countplot(x=data_train['Pclass_3'], hue=data_train['Survived'], palette='magma')

plt.ylim(0,700)

plt.legend(['Did not survive', 'Survived'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
num_trees = 1000

max_features = 3

kfold = KFold(n_splits=10, random_state=7)

rfc = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
type(y_train)
rfc.fit(X_train, y_train)
rfc.score(X_train, y_train) 
rfc.score(X_test, y_test)
y_pred = rfc.predict(X_test)
y_pred # this is an array of predictions
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
acc
type(y_test)
X_test.shape
y_test.shape
data_train.shape
data_test.shape
data_test.isnull().sum()
data_test.head()
np.array(y_test).reshape(-1,1).shape
data_train.shape
data_train.head()
X_train.head()
X_train.shape
y_train.head()
y_train.shape
X_test.head()
X_test.shape
y_test.head()
y_test.shape
data_test = data_test.drop(columns=['PassengerId'])
data_test.shape
df = pd.DataFrame({'PassengerId': range(892, 1310), 'Survived': (rfc.predict(data_test))})
type(df)
df.to_csv('TitanicDataSetKaggleVersion2.csv', index=False)
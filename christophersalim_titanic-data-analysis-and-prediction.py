# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from xgboost import XGBClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# load train and test data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
# view the head

train.head()
train.tail()
train.columns
print("There are {} rows in the training data".format(len(train)))

print("There are {} rows in the test data".format(len(test)))
train['train_test'] = 1

test['train_test'] = 0

test['Survived'] = np.NaN

all_data = pd.concat([train,test])
train.info()
train.describe()
train.sample(10)
train.isna().sum()
# separate numerical and categorical data because the way we visualize them will be different. 

train_num = train[['Age','SibSp','Parch','Fare']]

train_cat = train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]
train_num.hist(figsize=[14,14], grid=False)
# let's count the percentage of passengers with the age between 20-30

count_2030 = (train_num['Age'][(train_num['Age'] <= 30) & (train_num['Age'] >= 20)].count())/(train_num['Age'].count())

print(count_2030*100)

print("There are {:.2f} percent of people who were aged 20-30".format(count_2030*100))

# do the same for other 3 numerical features

count_zerosib = (train_num['SibSp'][(train_num['SibSp'] == 0)].count())/(train_num['Age'].count())

print("There are {:.2f} percent of people who had no siblings and spouse accompanying them".format(count_zerosib*100))



count_mostfare = (train_num['Fare'][(train_num['Fare'] <= 100) & (train_num['Fare'] >= 0)].count())/(train_num['Fare'].count())

print("There are {:.2f} percent of people who paid between 0 and 100 pounds".format(count_mostfare*100))



count_mostparch = (train_num['Parch'][(train_num['Parch'] == 0)].count())/(train_num['Parch'].count())

print("There are {:.2f} percent of people who had no parents and children accompanying them".format(count_mostparch*100))
print("There were {} males and {} females in the cabin.".format(train_cat['Sex'].value_counts().male, train_cat['Sex'].value_counts().female))
for j in train_cat.columns:

    plt.figure()

    ax = train_cat[j].value_counts().plot(kind='bar', title=j)

    ax.set_ylabel('Count')

    ax.set_xlabel('Labels')
# We don't need the ticket and cabin features yet.

train_cat_plot = train_cat.drop(['Ticket', 'Cabin'], axis=1)

fig = plt.figure(figsize=[10,10])

# make plots using all other features

for i, j in enumerate(train_cat_plot.columns):

    fig.add_subplot(2,2, i+1)

    sns.barplot(x=j, y='Survived', data=train_cat_plot)

    plt.title('{} vs Survived Plot'.format(j))
# also plot the numerical data

train_num_plot = train_num = train[['SibSp','Parch', 'Survived']]

fig = plt.figure(figsize=[10,10])

# make plots using all other features

for i, j in enumerate(train_num_plot.columns):

    fig.add_subplot(2,2, i+1)

    sns.barplot(x=j, y='Survived', data=train_num_plot)

    plt.title('{} vs Survived Plot'.format(j))
#print percentages of females vs. males that survive

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)



#print percentage of people by Pclass that survived

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
all_data.isna().sum()
all_data.Age = all_data.Age.fillna(train.Age.median()) # we use median because of some outliers.

all_data.Fare = all_data.Fare.fillna(train.Fare.median())

all_data = all_data.dropna(subset=['Embarked'])
all_data.isna().sum()
# Now handling duplicates

all_data.duplicated().sum()
# drop passenger id

all_data = all_data.drop(['PassengerId'], axis=1)
all_data_corrplot = all_data.drop(['train_test'], axis=1)

fig, axs = plt.subplots(nrows=1, figsize=(13, 13))

sns.heatmap(all_data_corrplot.corr(), annot=True, square=True, cmap='YlGnBu', linewidths=2, linecolor='black', annot_kws={'size':12})
all_data['name_title'] = all_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
all_data['name_title'].unique()
all_data['name_title'].value_counts()
# Replacing less familiar names with more familiar names

all_data['name_title'] = all_data['name_title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer')

all_data['name_title'] = all_data['name_title'].replace(['Jonkheer', 'Master'], 'Master')

all_data['name_title'] = all_data['name_title'].replace(['Don', 'Sir', 'the Countess', 'Lady', 'Dona'], 'Royalty')

all_data['name_title'] = all_data['name_title'].replace(['Mme', 'Ms', 'Mrs'], 'Mrs')

all_data['name_title'] = all_data['name_title'].replace(['Mlle', 'Miss'], 'Miss')

  



# Imputing missing values with 0

all_data['name_title'] = all_data['name_title'].fillna(0)



all_data['name_title'].value_counts()
plt.figure(figsize=[10,10])

all_data['name_title'].value_counts().plot(kind='bar')

plt.title('Count of Titles')

plt.xlabel('Title')

plt.ylabel('Count')
all_data['Cabin'] = all_data['Cabin'].fillna('Unknown')

all_data['Deck'] = all_data['Cabin'].str.get(0)
plt.figure(figsize=[10, 10])

sns.barplot(x=all_data['Deck'], y='Survived', data=all_data)

plt.title('Deck Category vs Survived Plot')
all_data['ticket_numeric'] = all_data['Ticket'].apply(lambda x: 1 if x.isnumeric() else 0)
plt.figure(figsize=[10, 10])

sns.barplot(x=all_data['ticket_numeric'], y='Survived', data=all_data)

plt.title('Is the ticket numeric only? vs Survived Plot')
all_data['num_family_member'] = all_data['SibSp'] + all_data['Parch']
all_data.head()
all_data = all_data.drop(['Name','SibSp', 'Parch', 'Ticket', 'Cabin', 'train_test'], axis=1)
all_data.head()
scale = StandardScaler()

all_data_scaled = all_data.copy()

all_data_scaled[['Age','num_family_member','ticket_numeric','Fare']]= scale.fit_transform(all_data_scaled[['Age','num_family_member','ticket_numeric','Fare']])
all_data.dtypes
all_data_encoded = all_data_scaled

for j in ['Sex', 'Embarked', 'name_title', 'Deck']:

    all_data_encoded[j] = all_data_encoded[j].astype('category')

    all_data_encoded[j] = all_data_encoded[j].cat.codes
all_data_encoded.head()
train_data = all_data[:len(train)-2] # because we dropped 2 column back in embarked column missing value

test_data = all_data[len(train)-2:]

print("Dimension of train data is: ", train_data.shape)

print("Dimension of test data is: ", test_data.shape)
X_train_scaled = train_data.drop(['Survived'], axis=1)

y_train_scaled = train_data['Survived']

X_test_scaled = test_data.drop(['Survived'], axis=1)

y_test_scaled = test_data['Survived']
X_train_scaled.head()
# NB model

print("The CV mean score for Naive Bayes model is {:.3f}".format(cross_val_score(GaussianNB(), X_train_scaled, y_train_scaled).mean()))
# KNN model

print("The CV mean score for K Nearest Neighbor model is {:.3f}".format(cross_val_score(KNeighborsClassifier(), X_train_scaled, y_train_scaled).mean()))
# SVM model

print("The CV mean score for Support Vector Machine model is {:.3f}".format(cross_val_score(SVC(), X_train_scaled, y_train_scaled).mean()))
# XGBoost model

print("The CV mean score for XGBoost model is {:.3f}".format(cross_val_score(XGBClassifier(), X_train_scaled, y_train_scaled).mean()))
# RF model

print("The CV mean score for Random Forest model is {:.3f}".format(cross_val_score(RandomForestClassifier(), X_train_scaled, y_train_scaled).mean()))
xgb = XGBClassifier()

xgb.fit(X_train_scaled, y_train_scaled)
y_pred = xgb.predict(X_test_scaled).astype('int32')
print(y_pred.shape)
final_result = {'PassengerId': test['PassengerId'], 'yhat': y_pred}
final_dataframe = pd.DataFrame(final_result)
final_dataframe.to_csv('submission.csv')
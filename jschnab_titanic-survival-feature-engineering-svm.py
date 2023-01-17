#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
#we will use regular expression for passenger names
import re 
from sklearn import svm
from sklearn.model_selection import train_test_split

#import data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#we put both data frames in a list to modify all data easily
data = [train, test]

#data quick check
print(train.head())
print(train.info())
print('\n', test.head())
print(test.info())
#check the effect of passenger class on survival rate
print('Survival rate depending passenger class:')
print(train[['Pclass', 'Survived']].groupby('Pclass', as_index=False).mean(), '\n')

#check the effect of passenger sex on survival rate
print('Survival rate depending on passenger sex:')
print(train[['Sex', 'Survived']].groupby('Sex', as_index=False).mean(), '\n')

#check the effect of number of siblings and spouses on survival rate
print('Survival rate depending on number of siblings and spouses:')
print(train[['SibSp', 'Survived']].groupby('SibSp', as_index=False).mean(), '\n')

#check the effect of number of parents and children on survival rate
print('Survival rate depending on number of parents and children:')
print(train[['Parch', 'Survived']].groupby('Parch', as_index=False).mean(), '\n')
#fill missing Age values
for dataset in data:
    #calculate mean, standard deviation and number of missing values
    avg = dataset['Age'].mean()
    std = dataset['Age'].std()
    null_count = dataset['Age'].isnull().sum()
    
    #generate random ages centered around the mean
    random_list = np.random.randint(avg - std, avg + std, size=null_count)
    
    #replace missing values with random ages
    dataset['Age'][np.isnan(dataset['Age'])] = random_list
    dataset['Age'] = dataset['Age'].astype(int)

#group Age values into 5 categories and check effect on survival
train['CategoricalAge'] = pd.cut(train['Age'], 5)
age_survival = train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()

#display barplot of results
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 10))
ax1.bar([0, 1, 2, 3, 4], age_survival['Survived'])
ax1.tick_params(axis='x', color='white')
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xticklabels(list(age_survival['CategoricalAge'].astype(str)), fontsize=15)
ax1.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.set_ylim(0, .6)
ax1.set_xlabel("Age categories (years)", fontsize=18)
ax1.set_ylabel("Survival rate", fontsize=18)
ax1.set_title("Impact of passenger age on survival", fontsize=18)


#fill missing "Embarked" values and show effect on survival
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
embarked_survival = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
ax2.bar([0, 1, 2], embarked_survival['Survived'])
ax2.set_xticks([0, 1, 2])
ax2.tick_params(axis='x', color='white')
ax2.set_xticklabels(['Cherbourg', 'Queenstown', 'Southampton'], fontsize=16)
ax2.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.set_ylim(0, .6)
ax2.set_xlabel("Boarding location", fontsize=18)
ax2.set_ylabel("Survival rate", fontsize=18)
ax2.set_title("Impact of boarding location on survival", fontsize=18)
plt.show()
#put fare in categories and check effect on survival
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)
fare_survival = train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

#engineer a feature indicating if a passenger travels alone
for dataset in data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
for dataset in data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
alone_survival = train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

#engineer a feature containing titles
#function to get title from name
def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ''

#extract titles from name and put them in a new feature
for dataset in data:
    dataset['Title'] = dataset['Name'].apply(extract_title)

#replace rarer titles by "Rare"
for dataset in data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Jonkheer',
                                                'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
title_survival = train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

#plot data for the engineered features
fig, ax = plt.subplots(figsize=(10, 8))
ax.bar([0, 1, 2, 3], fare_survival['Survived'])
ax.tick_params(axis='x', color='white')
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(list(fare_survival['CategoricalFare'].astype(str)), fontsize=15)
ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylim(0, .6)
ax.set_xlabel("Fare category", fontsize=18)
ax.set_ylabel("Survival rate", fontsize=18)
ax.set_title("Impact of fare on survival", fontsize=18)
plt.show()

fig, ax = plt.subplots()
ax.bar([0, 1], alone_survival['Survived'])
ax.tick_params(axis='x', color='white')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not alone', 'Alone'], fontsize=15)
ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylim(0, .6)
ax.set_ylabel("Survival rate", fontsize=18)
ax.set_title("Impact of being alone on survival", fontsize=18)
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
ax.bar([0, 1, 2, 3, 4], title_survival['Survived'])
ax.tick_params(axis='x', color='white')
ax.set_xticklabels([''] + list(title_survival['Title']), fontsize=15)
ax.set_yticklabels(labels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_ylim(0, 0.9)
ax.set_xlabel("Title", fontsize=18)
ax.set_ylabel("Survival rate", fontsize=18)
ax.set_title("Impact of title on survival", fontsize=18)
plt.show()

for dataset in data:
    
    #mapping sex
    dataset['Sex'] = dataset['Sex'].map({'female':0, 'male':1})
    
    #mapping titles
    title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':3, 'Rare':5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    #mapping embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2})
    
    #mapping fare
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    #mapping age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4
    
#feature selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch', 'FamilySize'] 
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalFare', 'CategoricalAge'], axis=1)
test_no_id = test.drop(drop_elements, axis=1) #without passenger id

print('Training dataset after feature engineering:\n', train.head(), '\n')
print('Testing dataset after feature engineering:\n', test.head())
#prepare data
#X are features
X = train.iloc[:, 1:]

#Y are targets
Y = train.iloc[:, 0]

#split train data into training and cross-validation datasets
X_train, X_cv, Y_train, Y_cv = train_test_split(X, Y, test_size=0.25)

#perform grid search to find the best parameters

#loop over grid of parameters
def grid_search():
    #list of parameters
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    kernels = ['linear', 'rbf', 'sigmoid']
    
    #variables to store the results
    best_score = 0
    best_C = None
    best_gamma = None
    best_kernel = None
    
    for C in C_values:
        for gamma in gamma_values:
            for kernel in kernels:
                svc = svm.SVC(C=C, gamma=gamma, kernel=kernel)
                svc.fit(X_train, Y_train)
                score = svc.score(X_cv, Y_cv)
                
                if score > best_score:
                    best_score = score
                    best_C = C
                    best_gamma = gamma
                    best_kernel = kernel
                    
    print('Best parameters give {0:.4%} accuracy'.format(best_score))
    print('C = {0}\ngamma = {1}\nKernel = {2}'.format(best_C, best_gamma, best_kernel))
    
    return best_C, best_gamma, best_kernel
#perform grid search
C, gamma, kernel = grid_search()
svc = svm.SVC(C=C, gamma=gamma, kernel=kernel, probability=True)
svc.fit(X, Y)
prediction = svc.predict(test_no_id)[:, np.newaxis]
prediction = np.hstack((test['PassengerId'][:, np.newaxis], prediction))
output = pd.DataFrame({'PassengerId':prediction[:, 0], 'Survived':prediction[:, 1]})
print(output.head())
np.savetxt('submission.csv', output, header='PassengerId,Survived', comments='', delimiter=',', fmt='%d')

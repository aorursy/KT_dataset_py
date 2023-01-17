# Importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

pd.options.display.max_columns = 100

import matplotlib

matplotlib.style.use('ggplot')

import warnings

import numpy as np

dataset= pd.read_csv("../input/train.csv")
%matplotlib inline

import seaborn

seaborn.set()

dataset.describe()

dataset.sample(5)
#There are only 714 entries out of 891 for age. 

#Substituting for the missing data by using median

dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

dataset.describe()
# Visualizing survival by class

survived_class  = dataset[dataset['Survived'] == 1]['Pclass'].value_counts()

dead_class = dataset[dataset['Survived'] == 0]['Pclass'].value_counts()

df_class = pd.DataFrame([survived_class, dead_class])

df_class.index = ['Survived', 'Dead']

df_class.plot(kind = 'bar', stacked = True, figsize = (5,5), title = 'Statistics by class')



class1_survived = df_class.iloc[0,0]/df_class.iloc[:, 0].sum()*100

class2_survived = df_class.iloc[0,1]/df_class.iloc[:, 1].sum()*100

class3_survived = df_class.iloc[0,2]/df_class.iloc[:, 2].sum()*100



print("Percentage of class 1 that survived:", class1_survived, "%")

print("Percentage of class 2 that survived:", class2_survived, "%")

print("Percentage of class 3 that survived:", class3_survived, "%")



from IPython.display import display

display(df_class)

# Visualizing survival by gender

survived_sex  = dataset[dataset['Survived'] == 1]['Sex'].value_counts()

dead_sex = dataset[dataset['Survived'] == 0]['Sex'].value_counts()

df_sex = pd.DataFrame([survived_sex, dead_sex])

df_sex.index = ['Survived', 'Dead']

df_sex.plot(kind = 'bar', stacked = True, figsize = (5,5), title = 'Statistics by sex')



female_survived = df_sex.iloc[0,0]/df_sex.iloc[:, 0].sum()*100

male_survived = df_sex.iloc[0,1]/df_sex.iloc[:, 1].sum()*100



print("Percentage of male that survived:", male_survived, "%")

print("Percentage of female that survived:", female_survived, "%")



from IPython.display import display

display(df_sex)
# Visualizing survival by age

figure  = plt.figure(figsize = (15,8))

plt.hist([dataset[dataset['Survived'] == 1]['Age'], dataset[dataset['Survived'] == 0]['Age']], stacked = True, color = ['g', 'r'], 

        bins = 50, label = ['Survived', 'Dead'])

plt.xlabel('Age')

plt.ylabel('number of passengers survived')

plt.legend()
# Visualizing survival by fare

figure  = plt.figure(figsize = (15,8))

plt.hist([dataset[dataset['Survived'] == 1]['Fare'], dataset[dataset['Survived'] == 0]['Fare']], stacked = True, color = ['g', 'r'], 

        bins = 50, label = ['Survived', 'Dead'])

plt.xlabel('Age')

plt.ylabel('number of passengers survived')

plt.legend()
#comparing the survival stats for fare vs age

plt.figure(figsize=(15,8))

ax = plt.subplot()

ax.scatter(dataset[dataset['Survived']==1]['Age'],dataset[dataset['Survived']==1]['Fare'],c='green',s=40)

ax.scatter(dataset[dataset['Survived']==0]['Age'],dataset[dataset['Survived']==0]['Fare'],c='red', s=40)

ax.set_xlabel('Age')

ax.set_ylabel('Fare')

ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
# visualizing survival for embarkation site

survived_embark = dataset[dataset['Survived']==1]['Embarked'].value_counts()

dead_embark = dataset[dataset['Survived']==0]['Embarked'].value_counts()

df = pd.DataFrame([survived_embark,dead_embark])

df.index = ['Survived','Dead']

df.plot(kind='bar', stacked=True, figsize=(15,8))



embarked_s = df.iloc[0,0]/df.iloc[:,0].sum()*100

embarked_c = df.iloc[0,1]/df.iloc[:,1].sum()*100

embarked_q = df.iloc[0,2]/df.iloc[:,2].sum()*100



print("Number of people survived who embarked in S", embarked_s, "%")

print("Number of people survived who embarked in C", embarked_c, "%")

print("Number of people survived who embarked in Q", embarked_q, "%")



from IPython.display import display

display(df)
def status(feature):

    

    print ('Processing', feature, ':ok')

       
def get_combined_data():

    #reading training data

    train = pd.read_csv('../input/train.csv')

    #reading test data

    test = pd.read_csv('../input/test.csv')

    

    #extracting and then removing targets from the training data

    targets = train.Survived

    train.drop('Survived', 1, inplace = True)

    

    #merging test and training data for feature engineering

    combined = train.append(test)

    combined.reset_index(inplace = True)

    combined.drop('index', 1, inplace = True)

    

    return combined
combined = get_combined_data()
combined.shape
combined.sample(5)
def get_titles():

    

    global combined

    

    #extract titles from each name

    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

    

    Title_dictionary = {

                        "Capt":       "Officer",

                        "Col":        "Officer",

                        "Major":      "Officer",

                        "Jonkheer":   "Royalty",

                        "Don":        "Royalty",

                        "Sir" :       "Royalty",

                        "Dr":         "Officer",

                        "Rev":        "Officer",

                        "the Countess":"Royalty",

                        "Dona":       "Royalty",

                        "Mme":        "Mrs",

                        "Mlle":       "Miss",

                        "Ms":         "Mrs",

                        "Mr" :        "Mr",

                        "Mrs" :       "Mrs",

                        "Miss" :      "Miss",

                        "Master" :    "Master",

                        "Lady" :      "Royalty"

                        }

    

    #we map each title

    combined['Title'] = combined.Title.map(Title_dictionary)

get_titles()
combined.sample(5)

grouped_train = combined.head(891).groupby(['Sex', 'Pclass', 'Title'])

grouped_median_train = grouped_train.median()



grouped_test = combined.iloc[891:].groupby(['Sex', 'Pclass', 'Title'])

grouped_median_test = grouped_test.median()
grouped_median_train
grouped_median_test
def process_age():

    global combined

    

    def fillAges(row, grouped_median):

        if row['Sex'] == 'female' and row['Pclass'] == 1:

            if row['Title'] == 'Miss':

                return grouped_median.loc['female', 1, 'Miss']['Age']

            elif row['Title'] == 'Mrs':

                return grouped_median.loc['female', 1, 'Mrs']['Age']

            elif row['Title'] == 'Officer':

                return grouped_median.loc['female', 1, 'Officer']['Age']

            elif row['Title'] == 'Royalty':

                return grouped_median.loc['female', 1, 'Royalty']['Age']

            

        elif row['Sex'] == 'female' and row['Pclass'] == 2:

            if row['Title'] == 'Miss':

                return grouped_median.loc['female', 2, 'Miss']['Age']

            elif row['Title'] == 'Mrs':

                return grouped_median.loc['female', 2, 'Mrs']['Age']

            

        elif row['Sex'] == 'female' and row['Pclass'] == 3:

            if row['Title'] == 'Miss':

                return grouped_median.loc['female', 3, 'Miss']['Age']

            elif row['Title'] == 'Mrs':

                return grouped_median.loc['female', 3, 'Mrs']['Age']

            

        elif row['Sex']=='male' and row['Pclass'] == 1:

            if row['Title'] == 'Master':

                return grouped_median.loc['male', 1, 'Master']['Age']

            elif row['Title'] == 'Mr':

                return grouped_median.loc['male', 1, 'Mr']['Age']

            elif row['Title'] == 'Officer':

                return grouped_median.loc['male', 1, 'Officer']['Age']

            elif row['Title'] == 'Royalty':

                return grouped_median.loc['male', 1, 'Royalty']['Age']

            

        elif row['Sex']=='male' and row['Pclass'] == 2:

            if row['Title'] == 'Master':

                return grouped_median.loc['male', 2, 'Master']['Age']

            elif row['Title'] == 'Mr':

                return grouped_median.loc['male', 2, 'Mr']['Age']

            elif row['Title'] == 'Officer':

                return grouped_median.loc['male', 2, 'Officer']['Age']

            

        elif row['Sex']=='male' and row['Pclass'] == 3:

            if row['Title'] == 'Master':

                return grouped_median.loc['male', 3, 'Master']['Age']

            elif row['Title'] == 'Mr':

                return grouped_median.loc['male', 3, 'Mr']['Age']

            elif row['Title'] == 'Officer':

                return grouped_median.loc['male', 3, 'Officer']['Age']

            

    combined.head(891).Age  = combined.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age'])

                                                      else r['Age'], axis = 1)

    combined.iloc[891:].Age = combined.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age'])

                                                       else r['Age'], axis = 1)

    status('age')
process_age()
combined.info()
def process_names():

    

    global combined

    combined.drop('Name', axis = 1, inplace = True)

    

    #encoding in a dummy variable

    titles_dummies = pd.get_dummies(combined['Title'], prefix = 'Title')

    combined = pd.concat([combined, titles_dummies], axis = 1)

    

    #removing titles

    combined.drop('Title', axis = 1, inplace = True)

    

    status('names')
process_names()
combined.head()
def process_fares():

    combined.head(891).Fare.fillna(combined.head(891).Fare.mean(), inplace = True)

    combined.iloc[891:].Fare.fillna(combined.iloc[891:].Fare.mean(), inplace = True)

    

    status('fare')
process_fares()
def process_embarked():

    

    global combined

    combined.head(891).Embarked.fillna('S', inplace = True)

    combined.iloc[891:].Embarked.fillna('S', inplace = True)

    

    #dummy encoding

    embarked_dummies = pd.get_dummies(combined['Embarked'], prefix = 'Embarked')

    combined = pd.concat([combined, embarked_dummies], axis = 1)

    combined.drop('Embarked', axis = 1, inplace = True)

    

    status('embarked')
process_embarked()
def process_cabin():

    

    global combined

    

    #replacing missing cabin with U (unknown)

    combined.Cabin.fillna('U', inplace = True)

    

    #mapping each cabin with the cabin letter

    combined['Cabin'] = combined['Cabin'].map(lambda c:c[0])

    

    #dummy encoding

    cabin_dummies = pd.get_dummies(combined['Cabin'], prefix = 'Cabin')

    combined = pd.concat([combined, cabin_dummies], axis = 1)

    combined.drop('Cabin', axis = 1, inplace = True)

    

    status('cabin')
process_cabin()
combined.info()
def process_sex():

    

    global combined

    combined['Sex'] = combined['Sex'].map({'male': 1, 'female': 0})

    status('sex')
process_sex()
def process_pclass():

    

    global combined

    

    pclass_dummies = pd.get_dummies(combined['Pclass'], prefix = 'Pclass')

    combined = pd.concat([combined, pclass_dummies], axis = 1)

    combined.drop('Pclass', axis = 1, inplace = True)



    status('Pclass')
process_pclass()
def process_ticket():

    

    global combined

    def cleanTicket(ticket):

        ticket = ticket.replace('.', '')

        ticket = ticket.replace('/', '')

        ticket = ticket.split()

        ticket = map(lambda t: t.strip(), ticket)

        ticket = [t for t in ticket if not str(t).isdigit()]

        if len(ticket)>0:

            return ticket[0]

        else:

            return 'XXX'

        

    # extracting dummy variables from ticket

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)

    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix = 'Ticket')

    combined = pd.concat([combined, tickets_dummies], axis = 1)

    combined.drop('Ticket', axis = 1, inplace = True)

    

    status('ticket')

        
process_ticket()
def process_family():

    

    global combined

    #introduce feature called as family size

    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1

    

    #introduce features based on family size

    combined['Singleton'] = combined['FamilySize'].map(lambda s: 1 if s == 1 else 0)

    combined['SmallFamily'] = combined['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)

    combined['LargeFamily'] = combined['FamilySize'].map(lambda s: 1 if 5<=s else 0)

    

    status('family')
process_family()
combined.shape
combined.head()
combined.drop('PassengerId', axis = 1, inplace = True)
combined.sample()
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.cross_validation import cross_val_score
def compute_score(clf, X, y, scoring = 'accuracy'):

    xval = cross_val_score(clf, X, y, cv = 5, scoring = scoring)

    return np.mean(xval)
def recover_test_train():

    global combined

    train0 = pd.read_csv('../input/train.csv')

    

    target = train0.Survived

    train = combined.head(891)

    test = combined.iloc[891:]

    

    return train, test, target
train, test, target = recover_test_train()
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

clf = RandomForestClassifier(n_estimators = 50, max_features = 'sqrt')

clf = clf.fit(train, target)
features = pd.DataFrame()

features['feature'] = train.columns

features['importance'] = clf.feature_importances_

features.sort_values(by = ['importance'], ascending = True, inplace = True)

features.set_index('feature', inplace = True)
features.plot(kind = 'barh', figsize = (20,20))
model = SelectFromModel(clf, prefit = True)

train_reduced = model.transform(train)

train_reduced.shape
test_reduced = model.transform(test)

test_reduced.shape
#set to true if you want to run grid search again

run_gs = False



if run_gs:

    parameter_grid = {

                'max_depth' : [4,6,8],

                'n_estimators' : [50,10],

                'max_features' : ['sqrt', 'auto', 'log2'],

                'min_samples_split' : [1,3,10],

                'min_samples_leaf' : [1,3,10],

                'bootstrap' : [True, False]

    }

    forest = RandomForestClassifier()

    cross_validation = StratifiedKFold(targets, n_folds = 5)



    grid_search = GridSearchCV(forest, scoring = 'accuracy', param_grid = parameter_grid, cv = cross_validation)



    grid_search.fit(train, target)

    model = grid_search

    parameters = grid_search.best_params_



    print('Best score: {}', format(grid_search.best_score_))

    print('Best params: {}', format(grid_search.best_params_))



else:

    parameters = {'bootstrap' : False, 'min_samples_leaf' : 3, 'n_estimators' : 50, 

                  'min_samples_split' : 10, 'max_features' : 'sqrt', 'max_depth' : 6}

    model = RandomForestClassifier(**parameters)

    model.fit(train, target)
compute_score(model, train, target, scoring = 'accuracy')
output = model.predict(test).astype(int)

df_output = pd.DataFrame()

aux = pd.read_csv('../input/test.csv')

df_output['PassengerId'] = aux['PassengerId']

df_output['Survived'] = output

df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)
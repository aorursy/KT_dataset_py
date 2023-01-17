import re

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style('white')



from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



import random

import time

from datetime import datetime



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline



pd.options.display.max_rows = 99

pd.options.display.max_columns = 99



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
whole = train.append(test)

whole = whole[train.columns]

whole.drop(['Fare', 'Cabin', 'Embarked'], axis=1, inplace=True)

def get_title(string):

    return re.findall('\,\ (.*?)\.', string)[0]



def get_family_name(string):

    return string[:string.index(',')]



whole['Title'] = whole.Name.apply(get_title)

title_mapping = {

    'Dr': 'Mr',

    'Don': 'Mr',

    'Rev' : 'Mr',

    'Jonkheer' : 'Mr',

    'Major' : 'Mr',

    'Col' : 'Mr',

    'Capt' : 'Mr',

    'Sir' : 'Mr',

    'Ms' : 'Miss',

    'Mlle' : 'Miss',

    'Lady' : 'Mrs',

    'Mme' : 'Mrs',

    'the Countess' : 'Mrs',

    'Dona' : 'Mrs',

}

whole.Title.replace(title_mapping, inplace=True)

whole.loc[(whole.Title=='Dr') & (whole.Sex == 'female'), 'Title'] = 'Mrs'



whole['FamilyName'] = whole.Name.apply(get_family_name)

whole['FamilySize'] = whole['SibSp'] + whole['Parch']



whole.head(5)
ticket_freq = whole.Ticket.value_counts()

duplicated_tickets = ticket_freq[ticket_freq > 1]

duplicated_tickets.shape[0]
grouped = duplicated_tickets.sum()

ungrouped = whole.shape[0] - grouped



plt.figure(figsize=(6,6))

plt.rc('font', size=14)

plt.pie([ungrouped, grouped], explode=[0, .05], labels=['individual', 'grouped'], 

        autopct='%1.0f%%', startangle=82)

plt.title('Grouped People are 46% of the entire passengers')

plt.show()
duplicated_tickets.hist(bins=9, align='left')

plt.title('Group Size Histogram')

plt.show()
whole[whole.Ticket == duplicated_tickets.index[0]]
whole[whole.Ticket == duplicated_tickets.index[5]]
whole[whole.Ticket == duplicated_tickets.index[9]]
whole[whole.Ticket == duplicated_tickets.index[15]]
group_dict = {'ticket':[], 'size':[], 'color':[], 'survival':[]}



for ticket in duplicated_tickets.index:

    group_dict['ticket'].append(ticket)

    group_dict['size'].append((whole.Ticket == ticket).sum())

    

    # get group color: Homogeneous vs Heterogeneous

    uniq_fn = whole.loc[whole.Ticket == ticket, 'FamilyName'].nunique()

    if uniq_fn == 1:

        group_dict['color'].append('Homo')

    else:

        group_dict['color'].append('Hetero')

        

    # get survival rate: Survived(all survived), Perished(all perished), Mixed(some survived)

    non_null_count = len(whole[(whole.Ticket == ticket) & (whole.Survived.notnull())])

    survived = whole.loc[whole.Ticket == ticket, 'Survived'].sum()

    

    if non_null_count <= 1:

        group_dict['survival'].append('Mixed') # only one known value -> mixed

    elif survived == 0:

        group_dict['survival'].append('Perished')

    elif non_null_count == survived:

        group_dict['survival'].append('Survived')

    else:

        group_dict['survival'].append('Mixed')

        



group_df = pd.DataFrame(group_dict)

group_df.head(5)
_, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4))

sns.countplot(x='color', data=group_df, ax=ax0)

ax0.set_title('Counts per Group Color')

sns.countplot(x='survival', data=group_df, ax=ax1)

ax1.set_title('Counts per Group Survival')

plt.show()
plt.figure(figsize=(10, 5))

sns.violinplot(x='survival', y='size', data=group_df)

plt.title('Group Size')

plt.show()
whole['GroupSize'] = 1

for tick in duplicated_tickets.index:

    size = (whole.Ticket == tick).sum()

    whole.loc[whole.Ticket == tick, 'GroupSize'] = size

    

_, (ax0, ax1) = plt.subplots(1, 2, figsize=(14,5))



whole.groupby(by='FamilySize').Survived.agg(np.mean).plot.bar(ax=ax0)

ax0.set_title('Survival Rate per Family Size')



whole.groupby(by='GroupSize').Survived.agg(np.mean).plot.bar(ax=ax1)

ax1.set_title('Survival Rate per Companion Group Size')



plt.show()
whole['Grouped'] = (whole.GroupSize > 1).astype('int8')

sns.barplot(x='Sex', y='Survived', hue='Grouped', data=whole)

plt.title('Survival Rate')

plt.show()
# create two features

whole['GroupColor'] = whole.Ticket.apply(lambda ticket: group_dict['color'][group_dict['ticket'].index(ticket)] if ticket in group_dict['ticket'] else 'NoGroup')

whole['GroupSurvival'] = whole.Ticket.apply(lambda ticket: group_dict['survival'][group_dict['ticket'].index(ticket)] if ticket in group_dict['ticket'] else 'NoGroup')



# remove non-numeric columns

corr_df = whole.drop(['PassengerId', 'Name', 'Ticket', 'Title', 'FamilyName'], axis=1)



# numerize categorical columns

corr_df.Sex.replace({'male': 0, 'female': 1}, inplace=True)

corr_df.GroupColor.replace({'NoGroup': 0, 'Homo': 1, 'Hetero': 2}, inplace=True)

corr_df.GroupSurvival.replace({'NoGroup': -1, 'Perished': 0, 'Mixed': 1, 'Survived': 2}, inplace=True)



# Chunking GroupSize

# The correlation value is not changed when I chunk the group size or not, so comment them for just now

# It may be worth when we apply this in models



# cut_points = [-1, 0, 3, 7, 20]

# label_names = ['No', 'Small', 'Big', 'Huge']

# corr_df['GroupSize'] = pd.cut(corr_df['GroupSize'], cut_points, labels=label_names)

# corr_df['GroupSize'].replace({'No': 0, 'Small': 1, 'Big':2, 'Huge':3}, inplace=True)



# show the heatmap

plt.figure(figsize=(12,12))

sns.heatmap(corr_df.corr(), cmap='RdBu_r', annot=True, center=0.0, fmt='.2g')

plt.title('Correlations between features')

plt.show()
class TitanicPreprocessor:

    

    def __init__(self, train, test):

        # store the length of train dataset to split merget dataframe again

        self.train_len = train.shape[0]

        

        # merge two dataframes into one

        self.whole = train.append(test)

        self.whole = self.whole[train.columns]  # keep the column order

        

    @staticmethod

    def get_title(string):

        return re.findall('\,\ (.*?)\.', string)[0]

    

    def __create_features(self):

        '''Create new features: Title and FS(family size)'''

        # Now we use Title feature for only filling missing values of Age feature

        self.whole['Title'] = self.whole.Name.apply(self.get_title)

        

        title_mapping = {

            'Dr': 'Mr',

            'Don': 'RareMale',

            'Rev' : 'RareMale',

            'Jonkheer' : 'RareMale',

            'Major' : 'RareMale',

            'Col' : 'RareMale',

            'Capt' : 'RareMale',

            'Sir' : 'Mr',

            'Ms' : 'RareFemale',

            'Mlle' : 'RareFemale',

            'Lady' : 'RareFemale',

            'Mme' : 'RareFemale',

            'the Countess' : 'RareFemale',

            'Dona' : 'RareFemale',

        }

        self.whole.Title.replace(title_mapping, inplace=True)

        

        # There is one female whose Title is Dr

        self.whole.loc[(self.whole.Title=='Dr') & (self.whole.Sex == 'female'), 'Title'] = 'Mrs'



        # Family Size

        self.whole['FS'] = self.whole['SibSp'] + self.whole['Parch']



    def __fill_missing_values(self):

        '''Deal with missing values(Age, Embarked, Fare, and Cabin)'''

        # Fill Age with the average age of the same class and title group.

        for index, row in self.whole[self.whole.Age.isnull()].iterrows():

            avg = self.whole.loc[(self.whole['Pclass'] == row['Pclass']) & (self.whole['Title'] == row['Title']), 'Age'].mean()

            if np.isnan(avg): # for rare case, there is no sample for the average of age.

                avg = self.whole.loc[self.whole['Title'] == row['Title'], 'Age'].mean()

            self.whole.loc[index, 'Age'] = avg

            

        # Fill Embarked and Fare with average values

        self.whole.Embarked.fillna(self.whole.Embarked.mode()[0], inplace=True)

        self.whole.Fare.fillna(self.whole.Fare.mean(), inplace=True)

        

        # We are not interested at Cabin which has so many missing values

        self.whole.drop('Cabin', axis=1, inplace=True)

        

    def __get_group_features(self):

        '''

        This method is to implement the proposed features 

        '''

        self.whole['FamilyName'] = self.whole.Name.apply(lambda x: x[:x.index(',')])



        ticket_freq = self.whole.Ticket.value_counts()

        duplicated_tickets = ticket_freq[ticket_freq > 1]        



        group_dict = {'ticket':[], 'size':[], 'color':[], 'survival':[]}



        for ticket in duplicated_tickets.index:

            group_dict['ticket'].append(ticket)

            group_dict['size'].append((self.whole.Ticket == ticket).sum())



            # get group color: Homogeneous vs Heterogeneous

            uniq_fn = self.whole.loc[self.whole.Ticket == ticket, 'FamilyName'].nunique()

            if uniq_fn == 1:

                group_dict['color'].append('Homo')

            else:

                group_dict['color'].append('Hetero')



            # get survival rate: Survived(all survived), Perished(all perished), Mixed(some survived)

            non_null_count = len(self.whole[(self.whole.Ticket == ticket) & (self.whole.Survived.notnull())])

            survived = self.whole.loc[self.whole.Ticket == ticket, 'Survived'].sum()



            if non_null_count <= 1:

                group_dict['survival'].append('Mixed') # only one known value -> mixed

            elif survived == 0:

                group_dict['survival'].append('Perished')

            elif non_null_count == survived:

                group_dict['survival'].append('Survived')

            else:

                group_dict['survival'].append('Mixed')





        self.whole['GroupColor'] = self.whole.Ticket.apply(lambda ticket: group_dict['color'][group_dict['ticket'].index(ticket)] if ticket in group_dict['ticket'] else 'NoGroup')

        self.whole['GroupSurvival'] = self.whole.Ticket.apply(lambda ticket: group_dict['survival'][group_dict['ticket'].index(ticket)] if ticket in group_dict['ticket'] else 'NoGroup')

        

        group_df = pd.DataFrame(group_dict)



        self.whole['GroupSize'] = 1

        for tick in duplicated_tickets.index:

            size = (self.whole.Ticket == tick).sum()

            self.whole.loc[self.whole.Ticket == tick, 'GroupSize'] = size

            

        self.whole['Grouped'] = (self.whole.GroupSize > 1).astype('int8')

        self.whole.GroupColor.replace({'NoGroup': 0, 'Homo': 1, 'Hetero': 2}, inplace=True)

        self.whole.GroupSurvival.replace({'NoGroup': -1, 'Perished': 0, 'Mixed': 1, 'Survived': 2}, inplace=True)

        

        cut_points = [-1, 0, 3, 7, 20]

        label_names = ['No', 'Small', 'Big', 'Huge']

        self.whole['GroupSize'] = pd.cut(self.whole['GroupSize'], cut_points, labels=label_names)

        self.whole['GroupSize'].replace({'No': 0, 'Small': 1, 'Big':2, 'Huge':3}, inplace=True)

    



    def __encoding(self):

        '''Converting categorical features into numerical ones.'''

        label = LabelEncoder()



        # Divide Fare into 3 groups and Age into 4 groups

        self.whole['FareCode'] = label.fit_transform(pd.qcut(self.whole.Fare, 3))

        self.whole['AgeCode'] = label.fit_transform(pd.qcut(self.whole.Age, 4))

        

        self.whole.Sex.replace({'male': 0, 'female': 1}, inplace=True)

        self.whole = pd.concat([self.whole, pd.get_dummies(self.whole.Title, prefix='Title')], axis=1)



    def get_processed(self):

        

        '''Processing all the feature engineering'''

        

        # feature engineering processes

        self.__create_features()

        self.__fill_missing_values()

        # new step for the proposed featuers

        self.__get_group_features()

        self.__encoding()

        

        # drop useless features

        self.whole.drop(['Name', 'Ticket', 'Embarked', 'Title', 'FamilyName'], axis=1, inplace=True)

        

        # get dataframes

        features = self.whole.columns.tolist()

        target = 'Survived'

        features.remove(target)

        features.remove('PassengerId')



        X = self.whole[:self.train_len][features]

        y = self.whole[:self.train_len]['Survived']

        test_X = self.whole[self.train_len:][features]

        

        # scaling

        std_scaler = StandardScaler()

        X = std_scaler.fit_transform(X)

        test_X = std_scaler.transform(test_X)

        

        return (X, y, test_X, features)



        

def fit_and_predict(model, train_X, train_y, test_X):

    # We got average score and average prediction

    score_sum = 0

    predict_sum = np.zeros(len(test_X))



    folds = 5

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)





    # For each fold

    for train_index, test_index in skf.split(train_X, train_y):

        X_train, X_test = X[train_index], X[test_index]

        y_train, y_test = y[train_index], y[test_index]    



        # fit & predict

        model.fit(X_train, y_train)

        predict = model.predict(X_test)



        # this is not a score but an error rate(MAE)

        score = np.sum(np.abs(predict - y_test)) / len(predict)

        score_sum += score



        # prediction averaging is a great help

        predict = model.predict(test_X)

        predict_sum += predict

        

    # average the results from folds

    avg_score = 1 - (score_sum / folds)

    avg_predict = predict_sum / folds

    

    return (avg_score, avg_predict)



def draw_feature_importance(model, features):

    df = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})

    df.sort_values(by='importance', ascending=True, inplace=True)

    df.plot(x='feature', y='importance', kind='barh')

    plt.title('Feature Importance')

    plt.show()
pp = TitanicPreprocessor(train, test)

X, y, test_X, features = pp.get_processed()



# We got this model from GridCV

model = RandomForestClassifier(criterion = 'entropy', 

                               max_depth = 10, 

                               max_features = 'log2', 

                               min_samples_leaf = 1, 

                               min_samples_split = 5, 

                               n_estimators = 6, 

                               random_state = 9)



score, pred = fit_and_predict(model, X, y, test_X)

print('CV Score:', score)

draw_feature_importance(model, features)



# submit the result

result_df = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred})

result_df.Survived = result_df.Survived.astype('uint8')

result_df.to_csv('submission-{0}-{1:.3f}.csv'.format(datetime.now().strftime('%y%m%d-%H%M'), score), index=False)
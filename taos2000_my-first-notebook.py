#import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import string
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV, cross_val_predict
#import data
dt_train = pd.read_csv('/kaggle/input/titanic/train.csv')#training set
dt_test = pd.read_csv('/kaggle/input/titanic/test.csv')#test set
dt_all = pd.concat([dt_train, dt_test], sort = True).reset_index(drop = True)#together
dt_combine = [dt_train, dt_test]
print('Training data shape = {}'.format(dt_train.shape))
print('Testing data shape = {}'.format(dt_test.shape))
print('ALL data shape = {}'.format(dt_all.shape))
print('Training data column = {}'.format(dt_train.columns.tolist()))
print('Testing data column = {}'.format(dt_test.columns.tolist()))
#Train data has 891 rows, test data has 418 rows
print(dt_train.info())#training dataset infomation
dt_train.head(5)
print(dt_test.info())#training dataset infomation
dt_test.sample(5)#5 random sample
#only three graphs, ctrl + c, ctrl + v is faster
fig, axs = plt.subplots(ncols=2, nrows=2)
plt.subplots_adjust(right=1.5, top=1.25)
plt.subplot(2,2,1)
sns.countplot(x = 'Pclass', hue = 'Survived', data = dt_train)
plt.subplot(2,2,2)
sns.countplot(x = 'Sex', hue = 'Survived', data = dt_train)
plt.subplot(2,2,3)
sns.countplot(x = 'Embarked', hue = 'Survived', data = dt_train)
plt.show()
# explore correlations between existing variables
# the correlations only includes non-categorical variables
sns.heatmap(dt_all.corr(), annot = True)
dt_all.corr()
#number of missing value(you can try feature engineering for categorical data before missing value)
dt_all.isnull().sum().sort_values(ascending = False)
#set expand to false. If True, return DataFrame with one column per capture group.
dt_all['Name_Title'] = dt_all.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
unique_title = dt_all ['Name_Title'].unique().tolist()
print(unique_title)
fig, axs = plt.subplots(figsize=(22, 9))
sns.countplot(x='Name_Title', hue='Survived', data=dt_all)

plt.xlabel('Name_Title', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=10)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Fare'), size=15, y=1.05)

plt.show()
dt_all['Name_Title'].value_counts(normalize = True) * 100
# Bundle rare salutations: 'Other' category
del unique_title[0:4]
dt_all['Name_Title'] = dt_all['Name_Title'].replace(unique_title, 'Other')
dt_all['Name_Title'].value_counts(normalize = True) * 100
new_category = {'Mr':1, 'Miss':2, 'Mrs':2, 'Master':3, 'Other':4}
dt_all['Name_Title'] = dt_all['Name_Title'].map(new_category)
dt_all['Name_Title'].unique().tolist()
dt_all['family'] = dt_all.Name.str.extract('([A-Za-z]+)\,', expand = False)
dt_all.head()
print(len(dt_all.family.unique().tolist()))
#survive: green not survive: red
survive = dt_train['Survived'] == 1
sns.distplot(dt_train[survive]['Age'].dropna(), label='Survived', hist=True, color='#e74c3c')
sns.distplot(dt_train[~survive]['Age'].dropna(), label='Survived', hist=True, color='#2ecc71')
dt_all['Age'] = pd.qcut(dt_all['Age'], 5)
sns.countplot(x='Age', hue='Survived', data=dt_all)
dt_all['Age'].head(5)
dt_all['Family_Size'] = dt_all['SibSp'] + dt_all['Parch'] + 1
sns.countplot(x = 'Family_Size', hue = 'Survived', data = dt_all)
#by the plot above, grouping 1 as group1, 2,3,4 as group2, 5,6,7 as group3, 8,11 as group4 (since they looks similar)
family_grouping = {1: 'group1', 2: 'group2', 3: 'group2', 4: 'group2', 5: 'group3', 6: 'group3', 7: 'group3', 8: 'group4', 11: 'group4'}
dt_all['Family_Size_Grouped'] = dt_all['Family_Size'].map(family_grouping)
sns.countplot(x = 'Family_Size_Grouped', hue = 'Survived', data = dt_all)
#survive: green not survive: red
plt.figure(figsize=(20,5))
survive = dt_train['Survived'] == 1
sns.distplot(dt_train[survive]['Fare'].dropna(), label='Survived', hist=True, color='#e74c3c')
sns.distplot(dt_train[~survive]['Fare'].dropna(), label='Survived', hist=True, color='#2ecc71')
plt.show()
#fill missing value before qcut or ccut. look at the data
dt_all[dt_all['Fare'].isnull()]
dt_all.corr()
print(dt_all.groupby(['Pclass','Family_Size','Name_Title']).Fare.median())
med_fare = dt_all.groupby(['Pclass', 'Family_Size', 'Name_Title']).Fare.median()[3][1][1]
#dt_all['Fare'] = dt_all['Fare'].fillna(med_fare)
plt.figure(figsize=(20,5))
dt_all['Fare'] = pd.qcut(dt_all['Fare'], 12)
sns.countplot(x='Fare', hue='Survived', data=dt_all)
# use M represent missing values
dt_all['Cabin'] = dt_all['Cabin'].fillna('M')
dt_all['Cabin'].unique().tolist()
#there might be one preson order one or more tickets. The first letter is unique. 
# Extract first letter
import re
#pandas.Series.map
dt_all['Cabin'] = dt_all['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group()) 
dt_all['Cabin'].unique().tolist()
type(dt_all['Cabin'])
#since it is a series type
dt_all['Cabin'].value_counts()
sort_list = ['M', 'C', 'E', 'G', 'D', 'A', 'B', 'F', 'T']
sort_list = sorted(sort_list)
print(sort_list)
#the first letter of tickets might relate to passenger class, according to daily experience.

dt_all_decks = dt_all.groupby(['Cabin','Pclass']).count()['Survived']
total_num_dict = {}
for letter in sort_list:
    total_num = 0
    for num in range(1,4):
        try:
            num_letter = dt_all_decks[letter][num]
            total_num += num_letter
            #print(total_num)
            if num == 3:
                total_num_dict[letter] = total_num
                total_num = 0
        except KeyError:
            if num == 3:
                total_num_dict[letter] = total_num
                total_num = 0
print(total_num_dict)                
print(dt_all_decks['A'][1])
print(dt_all_decks)
type(dt_all_decks)
precentage_num_dict = {}
for letter in sort_list:
    lis = []
    total = total_num_dict[letter]
    for num in range(1,4):
        try:
            lis.append(dt_all_decks[letter][num]/total)
        except KeyError:
            lis.append(0)
        if num == 3:
            precentage_num_dict[letter] = lis
print(precentage_num_dict)
#precentage chart
pd.DataFrame.from_dict(precentage_num_dict)
dt_all['Cabin'] = dt_all['Cabin'].replace(['A', 'B', 'C', 'T'], 'A&B&C&T')
dt_all['Cabin'] = dt_all['Cabin'].replace(['D', 'E'], 'D&E')
dt_all['Cabin'] = dt_all['Cabin'].replace(['F', 'G'], 'F&G')
dt_all['Cabin'].value_counts()
dt_all.head(5)
dt_all.drop(['Name', 'Family_Size'],axis=1,inplace=True)
dt_all.head(5)
dt_all['Ticket_Frequency'] = dt_all.groupby('Ticket')['Ticket'].transform('count')
dt_all.head(5)
#number of missing value
dt_all.isnull().sum().sort_values(ascending = False)
#label encoding, ignore NAN value
#https://stackoverflow.com/questions/54444260/labelencoder-that-keeps-missing-values-as-nan
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html
dt_all = dt_all.apply(lambda series: pd.Series(
    LabelEncoder().fit_transform(series[series.notnull()]),
    index=series[series.notnull()].index))
#number of missing value
dt_all.isnull().sum().sort_values(ascending = False)
dt_all.head(5)
dt_all[dt_all['Fare'].isnull()]
dt_all[dt_all['Embarked'].isnull()]
sns.heatmap(dt_all.corr(), annot = True)
dt_all.corr()
sort_age = dt_all.corr()['Age'].abs().sort_values(ascending = False)[1:]
#sort_age_fit = sort_age[sort_age > 0.1][1:]#drop the sane one
sort_age
sort_Embarked = dt_all.corr()['Embarked'].abs().sort_values(ascending = False)[1:]
#sort_Embarked_fit = sort_Embarked[sort_Embarked > 0.1][1:]
sort_Embarked
sort_Fare = dt_all.corr()['Fare'].abs().sort_values(ascending = False)[1:]
#sort_Fare_fit = sort_Fare[sort_Fare > 0.1][1:]
sort_Fare
sort_age_name = sort_age.index.tolist()
sort_Embarked_name = sort_Embarked.index.tolist()
sort_Fare_name = sort_Fare.index.tolist()
sort_lis = [sort_age_name,sort_Embarked_name, sort_Fare_name ]
for i in range(0,3):
    try:
        sort_lis[i].remove('Survived')
        sort_lis[i].remove('PassengerId')
    except:
        sort_lis[i] = sort_lis[i]
print('Age: ',sort_age_name)
print('Embarked: ',sort_Embarked_name)
print('Fare: ',sort_Fare_name)
#sort_Embarked_name.remove('Age')
dt_all['Fare'] = dt_all.groupby(sort_Fare_name[0])['Fare'].apply(lambda x: x.fillna(x.median()))
dt_all['Embarked'] = dt_all.groupby(sort_Embarked_name[0:2])['Embarked'].apply(lambda x: x.fillna(x.median()))
dt_all['Age'] = dt_all.groupby(['Sex',sort_age_name[0]])['Age'].apply(lambda x: x.fillna(x.median()))
dt_all.isnull().sum().sort_values(ascending = False)
fare_null = dt_all.loc[dt_all['PassengerId'] == 1043]
embark_null_1 = dt_all.loc[dt_all['PassengerId'] == 61]
embark_null_2 = dt_all.loc[dt_all['PassengerId'] == 829]
print(fare_null)
print(embark_null_1)
print(embark_null_2)
dt_all.columns
sort_Survived = dt_all.corr()['Survived'].abs().sort_values(ascending = False)[1:]
#sort_Fare_fit = sort_Fare[sort_Fare > 0.1][1:]
sort_Fare
dt_all.drop(['SibSp','Parch','family'], axis = 1, inplace = True)
dt_all.columns
OHEncoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
survive_all = dt_all[['Survived','PassengerId']]
dt_all_copy = dt_all.copy()
dt_all_copy = dt_all_copy.drop(['Survived','PassengerId'], axis = 1)
dt_all_copy_column = dt_all_copy.columns.tolist()
low_cardinality_cols = [col for col in dt_all_copy_column if dt_all_copy[col].nunique() < 5]
high_cardinality_cols = [col for col in dt_all_copy_column if dt_all_copy[col].nunique() >= 5]
print(low_cardinality_cols,high_cardinality_cols)
OH_cols_all = pd.DataFrame(OHEncoder.fit_transform(dt_all_copy[low_cardinality_cols]))
dt_all_onehog = pd.concat([OH_cols_all,survive_all,dt_all_copy[high_cardinality_cols]], axis = 1)
print(dt_all_onehog.shape)
dt_all_onehog.head()
dt_train_onehog = dt_all_onehog.loc[:890]
dt_test_onehog = dt_all_onehog.loc[891:]
dt_test_onehog.drop(['Survived'], axis = 1, inplace = True)
dt_train_onehog.head()
train_X_onehog = StandardScaler().fit_transform(dt_train_onehog.drop(['Survived', 'PassengerId'],axis = 1))
train_y_onehog = dt_train_onehog['Survived']
test_X_onehog = StandardScaler().fit_transform(dt_test_onehog.drop(['PassengerId'], axis = 1))
print(type(train_X_onehog))
print('train_X shape: {}'.format(train_X_onehog.shape))
print('train_y shape: {}'.format(train_y_onehog.shape))
print('test_X shape: {}'.format(test_X_onehog.shape))
forest_classifier = RandomForestClassifier(random_state=42)
'''
#this runs for a long time
forest_parameter = [
    {'max_depth':[2,4,7,8,12,16,20],
    'min_samples_split':[2,4,6,8,10],
    'min_samples_leaf':[2,4,6,8,10],
    'n_estimators':range(50,2000,50)}
]
'''
forest_parameter = [
    {'max_depth':[6],
    'min_samples_split':[6],
    'min_samples_leaf':[6],
    'n_estimators':range(50,2000,50)}]
ramdom_forest_model = GridSearchCV(forest_classifier, forest_parameter, cv=5, verbose=1, n_jobs=-1)
ramdom_forest_model.fit(train_X_onehog, train_y_onehog)
forest_best = ramdom_forest_model.best_estimator_
forest_best.fit(train_X_onehog, train_y_onehog)
forest_best_prediction = cross_val_predict(forest_best, train_X_onehog, train_y_onehog, cv=10)
forest_best_accuracy_score = accuracy_score(train_y_onehog, forest_best_prediction)
print(forest_best_accuracy_score)
forest_test_prediction = forest_best.predict(test_X_onehog)
forest_test_prediction = forest_test_prediction.astype(int)
random_forest_new_submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
random_forest_new_submission['PassengerId'] = dt_test_onehog['PassengerId'].add(1).to_numpy()
random_forest_new_submission['Survived'] = forest_test_prediction
random_forest_new_submission.to_csv('forest_submission_20200602.csv', header=True, index=False)
random_forest_new_submission.tail(5)
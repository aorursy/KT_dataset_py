import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math

%matplotlib inline



plt.rc("font", size=18)

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
print("Train observations", train.shape[0])

print("Test observations", test.shape[0])

print("Test Size", "{:.0%}".format(test.shape[0]/(train.shape[0]+test.shape[0])))
train.head()
train.info()

train.isnull().sum()
train.describe()
fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,5))

male = train[train.Sex == 'male']

female = train[train.Sex == 'female']

sns.distplot(female[female.Survived == 1].Age.dropna(), bins = 20,

             label = 'Survived', ax = ax[0], kde=False)

sns.distplot(female[female.Survived == 0].Age.dropna(), bins = 20,

             label = 'Not Survived', ax = ax[0], kde=False)

ax[0].legend()

ax[0].set_title('Female')

sns.distplot(male[male.Survived == 1].Age.dropna(), bins = 20,

             label = 'Survived', ax = ax[1], kde=False)

sns.distplot(male[male.Survived == 0].Age.dropna(), bins = 20,

             label = 'Not Survived', ax = ax[1], kde=False)

ax[1].legend()

ax[1].set_title('Male')



print("Total count of Male surviors", male[male.Survived == 1].shape[0], 

      "| Percentage of Total Male", "{:.0%}".format(male[male.Survived == 1].shape[0]/male.shape[0]))

print("Total count of Female surviors", female[female.Survived == 1].shape[0], 

      "| Percentage of Total Female", "{:.0%}".format(female[female.Survived == 1].shape[0]/female.shape[0]))
from numpy import mean

fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (10,5))

sns.barplot(x = 'SibSp', y = 'Survived', data = train, ci=None, color='salmon', estimator = mean, ax = ax[0])

ax[0].set_title('Mean Survival of Siblings or Spouse')

sns.barplot(x = 'Parch', y = 'Survived', data = train, ci=None, color = 'indigo', estimator = mean, ax = ax[1])

ax[1].set_title('Mean Survival of Parents or Children')
def create_matrix(variable1, variable2):

    if max(train[variable1].unique()) > max(train[variable2].unique()):

        number = max(train[variable1].unique())+1

    else:

        number = max(train[variable2].unique())+1

    

    matrix_data = np.array([[np.empty for i in range(number)] for j in range(number)])

    for i in range(number):

        for j in range(number):

            matrix_data[i, j] = 'child or parent'

            matrix_data[6, j] = ''

            matrix_data[7, j] = ''

            matrix_data[0, 0] = 'adult'

            matrix_data[1, 0] = 'adult'

            if i>1:

                matrix_data[i, j] = 'child'

            if j>2:

                matrix_data[i, j] = 'parent'

            if j ==7 or 8:

                matrix_data[i, 7] = ''

                matrix_data[i, 8] = ''

    

    for i in range(number):

        for j in range(number):

            if j not in train[train[variable1] == i][variable2].unique():

                matrix_data[i, j] = ''

                

    columns = [variable2+ ' ' + str(i) for i in range(number)]

    matrix = pd.DataFrame(data = matrix_data, columns = columns)

    matrix[variable1] = [variable1+ ' ' + str(i) for i in range(number)]

    matrix = matrix.set_index(variable1)

    matrix = matrix.drop(['Parch 7', 'Parch 8'], axis =1)

    return matrix
create_matrix('SibSp', 'Parch')
def generate_persona(dataset, persona_list):

    for i in range(dataset.shape[0]):

        if dataset.Age[i] <=14:

            persona_list.append('child')

        if dataset.Age[i] > 14:

            if dataset.Parch[i] > 0:

                persona_list.append('parent')

            elif dataset.Parch[i] == 0:

                persona_list.append('adult')

        if math.isnan(dataset.Age[i]) == True:

            if dataset.SibSp[i] in [0, 1] and dataset.Parch[i] == 0:

                persona_list.append('adult')

            elif dataset.SibSp[i] >=2 and dataset.Parch[i] < 3:

                persona_list.append('child')

            elif dataset.Parch[i] >2:

                persona_list.append('parent')

            else:

                persona_list.append('child or parent')

    dataset['persona'] = persona_list
persona_list_train = []

persona_list_test = []

persona_lists = [persona_list_train, persona_list_test]

data = [train, test]

for i in range(2):

    generate_persona(data[i], persona_lists[i])
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,5))

sns.countplot(x = 'persona', data = train, color='salmon', hue = 'Sex', ax=ax)

ax.set_title('Count by Persona')
from numpy import sum

fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,5))

sns.barplot(x = 'persona', y = 'Survived', data = train[train.Survived ==1], ci=None, color='salmon', hue = 'Sex', estimator = sum, ax=ax)

ax.set_title('Count Survival by Persona')
fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (10,5))

sns.barplot(x = 'persona', y = 'Survived', data = train, ci=None, color='indigo', hue = 'Sex', estimator = mean, ax=ax)

ax.set_title('Mean Survival by Persona')
fig, ax = plt.subplots(nrows=2, ncols=2, figsize = (12,7))

sns.countplot(x = 'Embarked', data = train, color='salmon', hue = 'persona', ax = ax[0,0])

ax[0,0].set_title('Count by Embarking Point')

sns.countplot(x = 'Pclass', data = train, color = 'indigo', hue = 'persona', ax = ax[0,1])

ax[0,1].set_title('Count by Passenger Class')

sns.barplot(x = 'Embarked', y = 'Survived', data = train, ci=None, color='salmon', 

            hue = 'persona', estimator = mean, ax = ax[1,0])

ax[1,0].set_title('Mean Survival by Embarking Point')

sns.barplot(x = 'Pclass', y = 'Survived', data = train, ci=None, color = 'indigo', hue = 'persona', 

            estimator = mean, ax = ax[1,1])

ax[1,1].set_title('Mean Survival by Passenger Class')

plt.tight_layout()
def generate_titles(dataset, titles_list):

    for i in dataset.Name:

        split = i.split(' ')

        for j in range(len(split)):

            if ',' in split[j]:

                if split[j+2] == 'Countess.':

                    titles_list.append(split[j+2])

                else:

                    titles_list.append(split[j+1])

    dataset['titles'] = titles_list
def generate_surnames(dataset, surname_list):

    for i in dataset.Name:

        surname_list.append(i.split(' ')[0][:-1])
train_titles = []

test_titles = []

titles_lists = [train_titles, test_titles]

data = [train, test]

for i in range(2):

    generate_titles(data[i], titles_lists[i])
create_matrix('SibSp', 'Parch')
train[train.persona == 'child'].titles.unique()
train[train.persona == 'parent'].titles.unique()
train[train.persona == 'adult'].titles.unique()
train[train.persona == 'child or parent'].titles.unique()
train[train.persona == 'child or parent']
train.loc[(train.persona == 'child or parent')&(train.titles == 'Master.'), 'persona'] = 'child'
train[train.persona == 'child or parent'].titles.unique()
train.isnull().sum()
train_df = train.copy()

test_df = test.copy()
train_df = train_df.drop(['Cabin'], axis=1)

test_df = test_df.drop(['Cabin'], axis = 1)
train_df[train_df.Embarked.isnull()]
train[train.Embarked.isnull()]
train_df.Embarked.describe()
replace_value = 'S'

data = [train_df, test_df]



for dataset in data:

    dataset['Embarked'] = dataset['Embarked'].fillna(replace_value)
train_df.persona.unique()
print("Null adult persona", train_df[train_df.persona == 'adult'].Age.isnull().sum())

print("Median age", train_df[train_df.persona == 'adult'].Age.median())

train_df[train_df.persona == 'adult'].describe()
train_df.loc[(train_df.Age.isnull())&(train_df.persona=='adult'),'Age']= train_df[train_df.persona == 'adult'].Age.median()
test_df.loc[(test_df.Age.isnull())&(test_df.persona=='adult'),'Age']= test_df[test_df.persona == 'adult'].Age.median()
print("Null child or parent persona", train_df[train_df.persona == 'child or parent'].Age.isnull().sum())

print("Median age", train_df[train_df.persona == 'child or parent'].Age.median())

train_df[train_df.persona == 'child or parent'].describe()
create_matrix('SibSp', 'Parch')
pd.set_option("display.max_rows", 1000)

train_df[train_df.persona == 'child or parent']
train_df.loc[(train_df.persona == 'child or parent')&(train.titles == 'Mrs.'), 'persona'] = 'parent'

train_df.loc[(train_df.persona == 'child or parent')&(train.titles == 'Miss.'), 'persona'] = 'child'

train_df.loc[(train_df.persona == 'child or parent')&(train.titles == 'Mr.'), 'persona'] = 'parent'
test_df.loc[(test_df.persona == 'child or parent')&(test.titles == 'Mrs.'), 'persona'] = 'parent'

test_df.loc[(test_df.persona == 'child or parent')&(test.titles == 'Master.'), 'persona'] = 'child'
print("Null child persona", train_df[train_df.persona == 'child'].Age.isnull().sum())

print("Median age", train_df[train_df.persona == 'child'].Age.median())

train_df[train_df.persona == 'child'].describe()
train_df.loc[(train_df.Age.isnull())&(train_df.persona=='child'),'Age']= train_df[train_df.persona == 'child'].Age.median()
test_df.loc[(test_df.Age.isnull())&(test_df.persona=='child'),'Age']= test_df[test_df.persona == 'child'].Age.median()
print("Null parent persona", train_df[train_df.persona == 'parent'].Age.isnull().sum())

print("Median age", train_df[train_df.persona == 'parent'].Age.median())

train_df[train_df.persona == 'parent'].describe()
train_df.loc[(train_df.Age.isnull())&(train_df.persona=='parent'),'Age']= train_df[train_df.persona == 'parent'].Age.median()
test_df.loc[(test_df.Age.isnull())&(test_df.persona=='parent'),'Age']= test_df[test_df.persona == 'parent'].Age.median()                                                                 
train_df.isnull().sum()
test_df.isnull().sum()
test_df.loc[test_df.Fare.isnull(), 'Fare'] = test_df.Age.mean()
test_df.isnull().sum()
train_df.head(2)
train_df.Sex = train_df.Sex.map({'female': 1, 'male': 0}).astype(int)
test_df.Sex = test_df.Sex.map({'female': 1, 'male': 0}).astype(int)
train_df.Embarked.unique()
train_df.Embarked = train_df.Embarked.map({'Q': 0, 'C': 1, 'S': 2}).astype(int)
test_df.Embarked = test_df.Embarked.map({'Q': 0, 'C': 1, 'S': 2}).astype(int)
train_df.persona.unique()
train_df.persona = train_df.persona.map({'adult': 0, 'parent': 1, 'child': 2}).astype(int)
test_df.persona = test_df.persona.map({'adult': 0, 'parent': 1, 'child': 2}).astype(int)
train_df.titles.unique()
train_df.titles = train_df.titles.replace(['Don.', 'Dr.', 'Major.', 'Lady.', 'Sir.',

                                       'Col.', 'Capt.', 'Countess.', 'Jonkheer.', 'Rev.'], 'Rare')

train_df.titles = train_df.titles.replace('Mlle.', 'Miss.')

train_df.titles = train_df.titles.replace('Ms.', 'Miss.')

train_df.titles = train_df.titles.replace('Mme.', 'Miss.')
test_df.titles = test_df.titles.replace(['Don.', 'Dr.', 'Major.', 'Lady.', 'Sir.',

                                       'Col.', 'Capt.', 'Countess.', 'Jonkheer.', 'Rev.'], 'Rare')

test_df.titles = test_df.titles.replace('Mlle.', 'Miss.')

test_df.titles = test_df.titles.replace('Ms.', 'Miss.')

test_df.titles = test_df.titles.replace('Mme.', 'Miss.')
train_df.titles.unique()
test_df.titles.unique()
test_df.titles = test_df.titles.replace('Dona.', 'Rare')
train_df.titles = train_df.titles.map({"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare": 5}).astype(int)
test_df.titles = test_df.titles.map({"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Rare": 5}).astype(int)
train_df.head(2)

test_df.head(2)
from sklearn import preprocessing

continuous_features = ['Fare','Age']

data = [train_df, test_df]

for dataset in data:

    for col in continuous_features:

        transf = dataset[col].values.reshape(-1,1)

        scaler = preprocessing.StandardScaler().fit(transf)

        dataset[col] = scaler.transform(transf)
train_df = pd.get_dummies(train_df, columns=['Embarked','titles','Parch','SibSp','Pclass', 'persona'], drop_first = False)

test_df = pd.get_dummies(test_df, columns=['Embarked','titles','Parch','SibSp','Pclass', 'persona'], drop_first = False)
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)

test_df = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
pd.set_option("display.max_columns", 40)

train_df.head(2)

test_df.head(2)
def get_gini_impurity(survived_count, total_count):

    survival_prob = survived_count/total_count

    not_survival_prob = (1 - survival_prob)

    random_observation_survived_prob = survival_prob

    random_observation_not_survived_prob = (1 - random_observation_survived_prob)

    mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob

    mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob

    gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob

    return gini_impurity
gini_impurity_starting_node = get_gini_impurity(342, 891)

gini_impurity_starting_node
gini_impurity_male = get_gini_impurity(109, 577)

print("male gini", gini_impurity_male)

gini_impurity_female = get_gini_impurity(233, 314)

print("female gini", gini_impurity_female)
#Weighted =impurity

men_weight = 577/891

women_weight = 314/891

weighted_gini_impurity_sex_split = (gini_impurity_male * men_weight) + (gini_impurity_female * women_weight)



sex_gini_decrease = weighted_gini_impurity_sex_split - gini_impurity_starting_node

sex_gini_decrease
print("adult observations", train_df[(train_df.persona_0 ==1)].shape[0])

print("adult survived", train_df[(train_df.persona_0 ==1)&(train_df.Survived == 1)].shape[0])
gini_impurity_adult = get_gini_impurity(226, 668)

print("adult gini", gini_impurity_adult)

gini_impurity_nonadult = get_gini_impurity((342-226), (891-668))

print("nonadult gini", gini_impurity_nonadult)
#Weighted =impurity

adult_weight = 668/891

nonadult_weight = (891-668)/891

weighted_gini_impurity_adult_split = (gini_impurity_adult * adult_weight) + (gini_impurity_nonadult * nonadult_weight)



adult_gini_decrease = weighted_gini_impurity_adult_split - gini_impurity_starting_node

adult_gini_decrease
print("child observations", train_df[(train_df.persona_2 ==1)].shape[0])

print("child survived", train_df[(train_df.persona_2 ==1)&(train_df.Survived == 1)].shape[0])

print("nonchild observations", train_df[(train_df.persona_2 ==1)].shape[0])

print("child survived", train_df[(train_df.persona_2 ==1)&(train_df.Survived == 1)].shape[0])
gini_impurity_child = get_gini_impurity(50, 96)

print("child gini", gini_impurity_child)

gini_impurity_nonchild = get_gini_impurity((342-50), (891-96))

print("non-child gini", gini_impurity_nonchild)
#Weighted =impurity

child_weight = 96/891

nonchild_weight = (891-96)/891

weighted_gini_impurity_child_split = (gini_impurity_child * child_weight) + (gini_impurity_nonchild * nonchild_weight)



child_gini_decrease = weighted_gini_impurity_child_split - gini_impurity_starting_node

child_gini_decrease
print("mr observations", train_df[(train_df.titles_1 ==1)].shape[0])

print("mr survived", train_df[(train_df.titles_1 ==1)&(train_df.Survived == 1)].shape[0])
gini_impurity_mr = get_gini_impurity(81, 517)

print("mr gini", gini_impurity_mr)

gini_impurity_nonmr = get_gini_impurity((342-81), (891-517))

print("non-mr gini", gini_impurity_nonmr)
#Weighted =impurity

mr_weight = 517/891

nonmr_weight = (891-517)/891

weighted_gini_impurity_mr_split = (gini_impurity_mr * mr_weight) + (gini_impurity_nonmr * nonmr_weight)

mr_gini_decrease = weighted_gini_impurity_mr_split - gini_impurity_starting_node

mr_gini_decrease
train_df.head(2)
def get_impurity_change(dataset, variable):

    

    def get_gini_impurity(survived_count, total_count):

        survival_prob = survived_count/total_count

        not_survival_prob = (1 - survival_prob)

        random_observation_survived_prob = survival_prob

        random_observation_not_survived_prob = (1 - random_observation_survived_prob)

        mislabelling_survided_prob = not_survival_prob * random_observation_survived_prob

        mislabelling_not_survided_prob = survival_prob * random_observation_not_survived_prob

        gini_impurity = mislabelling_survided_prob + mislabelling_not_survided_prob

        return gini_impurity

    

    obs = dataset[(dataset[variable] ==1)].shape[0]

    survived_obs = dataset[(dataset[variable] ==1)&(dataset.Survived == 1)].shape[0]

    total_obs = dataset.shape[0]

    total_surv = dataset[dataset.Survived == 1].shape[0]

    gini_impurity_1 = get_gini_impurity(survived_obs, obs)

    gini_impurity_2 = get_gini_impurity((total_surv-survived_obs), (total_obs - obs))

    gini_impurity_starting_node = get_gini_impurity(total_surv, total_obs)

    weight1 = obs/total_obs

    weight2 = (total_obs - obs)/total_obs

    weighted_gini = (gini_impurity_1 * weight1) + (gini_impurity_2 * weight2)

    gini_decrease = weighted_gini - gini_impurity_starting_node

    

    return gini_decrease
get_impurity_change(train_df, 'Sex')
select_columns = train_df.columns.to_list()

select_columns = select_columns[1:2] + select_columns[4:] #remove non binary variables
for i in select_columns:

    print(i, get_impurity_change(train_df, i))
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split



X_train = train_df.drop(['Survived'], axis = 1).values

y_train = train_df['Survived']

X_test = test_df.drop(['Parch_9'], axis = 1).values
decision_tree = DecisionTreeClassifier(max_depth = 3)

decision_tree.fit(X_train, y_train)
y_pred = decision_tree.predict(X_test)
from sklearn.metrics import accuracy_score

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
len(y_pred)
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = y_pred

submission = submission.set_index('PassengerId')
submission.to_csv('titanic_submission_03022020.csv')
from sklearn.model_selection import GridSearchCV
params = {'max_leaf_nodes': list(range(2, 100)), 

          'min_samples_split': [2, 3, 4], 

          'criterion':['entropy', 'gini']}

          

grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)



grid_search_cv.fit(X_train, y_train)
grid_search_cv.best_params_
tree_clf = DecisionTreeClassifier(criterion= 'entropy', max_leaf_nodes = 47, min_samples_split = 4, 

                                  random_state=42)
tree_clf.fit(X_train, y_train)
y_pred1 = tree_clf.predict(X_test)
acc_decision_tree = round(tree_clf.score(X_train, y_train) * 100, 2)

acc_decision_tree
submission2 = pd.DataFrame()

submission2['PassengerId'] = test['PassengerId']

submission2['Survived'] = y_pred1

submission2 = submission2.set_index('PassengerId')
submission2.to_csv('titanic_submission_03022020_sub2.csv')
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
params = {'bootstrap': [True],

          'max_depth': list(range(3, 11)), 

          'max_features': ["sqrt", "log2"], 

          'max_leaf_nodes':list(range(6,10))}



grid_search = GridSearchCV(model, params, 

                          verbose = 1)



grid_search.fit(X_train, y_train)
grid_search.best_params_
randomf_tree = RandomForestClassifier(bootstrap = True, max_depth = 9, max_features = 'log2', max_leaf_nodes =8,

                                     random_state = 0)
randomf_tree.fit(X_train, y_train)
y_pred2 = randomf_tree.predict(X_test)
acc_random_tree = round(randomf_tree.score(X_train, y_train) * 100, 2)

acc_random_tree
submission3 = pd.DataFrame()

submission3['PassengerId'] = test['PassengerId']

submission3['Survived'] = y_pred2

submission3 = submission3.set_index('PassengerId')
submission3.to_csv('titanic_submission_03022020_sub3.csv')
params = {'bootstrap': [True],

          'max_depth': list(range(3, 16)), 

          'max_features': ["auto", "sqrt", "log2"], 

          'max_leaf_nodes':list(range(6,20))}



grid_search = GridSearchCV(model, params, 

                          verbose = 1, cv =10)



grid_search.fit(X_train, y_train)
grid_search.best_params_
randomf_tree_new = RandomForestClassifier(bootstrap = True, max_depth = 9, max_features = 'sqrt', max_leaf_nodes =18)
randomf_tree_new.fit(X_train, y_train)
y_pred4 = randomf_tree_new.predict(X_test)
acc_random_tree_5 = round(randomf_tree_new.score(X_train, y_train) * 100, 2)

acc_random_tree_5
submission5 = pd.DataFrame()

submission5['PassengerId'] = test['PassengerId']

submission5['Survived'] = y_pred4

submission5 = submission5.set_index('PassengerId')
submission5.to_csv('titanic_submission_04022020_2.csv')
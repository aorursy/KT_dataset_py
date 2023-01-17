# Importing required libraries



import pandas as pd

import numpy as np

import re

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 20)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import sklearn

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
# Reading data into a dataframe

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train.info()

# can also use train.isnull().sum()
test.info()
train['Embarked'] = train['Embarked'].fillna(np.argmax(train['Embarked'].value_counts()))
train['Embarked'].unique()
age_combined = np.concatenate((test['Age'].dropna(), train['Age'].dropna()), axis=0)

mean = age_combined.mean()

std_dev = age_combined.std()

train_na = np.isnan(train['Age'])

test_na = np.isnan(test['Age'])

impute_age_train = np.random.randint(mean - std_dev, mean + std_dev, size = train_na.sum())

impute_age_test = np.random.randint(mean - std_dev, mean + std_dev, size = test_na.sum())

train["Age"][train_na] = impute_age_train

test["Age"][test_na] = impute_age_test

new_age_combined = np.concatenate((test["Age"],train["Age"]), axis = 0)
# Check the effect of imputation on the distribution

_ = sns.kdeplot(age_combined)

_ = sns.kdeplot(new_age_combined)
print(test['Fare'].isnull().sum())

test['Fare'] = test['Fare'].fillna(np.median(train['Fare']))

print(test['Fare'].isnull().sum())
train["Cabin"] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test["Cabin"] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
mean_survival_cabin = train[["Cabin", "Survived",'Sex']].groupby(['Cabin','Sex'],as_index=False).mean()

sns.set(font_scale=1.7)

ax = sns.barplot(x='Cabin', y='Survived', hue = 'Sex', data=mean_survival_cabin)

ax.legend(loc = 'upper left')
train['family_size'] = train['SibSp'] + train['Parch'] + 1

test['family_size'] = test['SibSp'] + test['Parch'] + 1

cols_to_drop = ['SibSp','Parch']

train.drop(cols_to_drop, inplace = True, axis = 1)

test.drop(cols_to_drop, inplace = True, axis = 1)
sns.factorplot('family_size','Survived', hue = 'Sex', data=train, size = 5, aspect = 3)
# Lets try to find a pattern in 'Name' column

train[['Name']].head(20)
titles_train = train['Name'].str.extract(' ([A-Za-z]+)\.')
print(titles_train.value_counts(),'\n')

print('Null value count is ', titles_train.isnull().sum())
titles_train.replace(['Countess', 'Dona', 'Lady', 'Mme'], 'Mrs', inplace = True)

titles_train.replace(['Mlle', 'Ms'], 'Miss', inplace = True)

rare_titles = []

temp = titles_train.value_counts()

for title in temp.index:

    if temp[title] < 10:

        rare_titles.append(title)

        

titles_train.replace(rare_titles, 'rare', inplace = True)
train['title'] = titles_train

del titles_train
# Repeat the procedure with test dataset.

titles_test = test['Name'].str.extract(' ([A-Za-z]+)\.')

print(titles_test.value_counts(),'\n')

print('Null value count is ', titles_test.isnull().sum())
titles_test.replace(['Countess', 'Dona', 'Lady', 'Mme'], 'Mrs', inplace = True)

titles_test.replace(['Mlle', 'Ms'], 'Miss', inplace = True)

rare_titles = []

temp = titles_test.value_counts()

for title in temp.index:

    if temp[title] < 10:

        rare_titles.append(title)

        

titles_test.replace(rare_titles, 'rare', inplace = True)

test['title'] = titles_test

del titles_test
test['title'].value_counts()
# put train and test datasets in one list for the ease of doing operations.

data = [train, test]



# delete 'Ticket' and 'Name' columns

for df in data:

    df.drop(['Ticket','Name'], inplace = True, axis = 1)
# 'Sex' column - straightforward 0 and 1 mapping

train['Sex'] = train['Sex'].apply(lambda x:1 if x == 'female' else 0)

test['Sex'] = test['Sex'].apply(lambda x:1 if x == 'female' else 0)
f, [ax1,ax2] = plt.subplots(1,2, figsize = (20,5))

sns.distplot(train['Age'][train['Survived'] == 1][train['Sex'] == 0], hist = False, ax = ax1, norm_hist = True, 

             label = 'Survived')

sns.distplot(train['Age'][train['Sex'] == 0], hist = False, ax = ax1, norm_hist = True, label = 'Male age distribution')

sns.distplot(train['Age'][train['Survived'] == 0][train['Sex'] == 0], hist = False, ax = ax2, norm_hist = True, 

             label = 'Didn\'t Survive')

sns.distplot(train['Age'][train['Sex'] == 0], hist = False, ax = ax2, norm_hist = True, label = 'Male age distribution')
f, [ax1,ax2] = plt.subplots(1,2, figsize = (20,5))

sns.distplot(train['Age'][train['Survived'] == 1][train['Sex'] == 1], hist = False, ax = ax1, norm_hist = True, 

             label = 'Survived')

sns.distplot(train['Age'][train['Sex'] == 1], hist = False, ax = ax1, norm_hist = True, label = 'Female age distribution')

sns.distplot(train['Age'][train['Survived'] == 0][train['Sex'] == 1], hist = False, ax = ax2, norm_hist = True, 

             label = 'Didn\'t Survive')

sns.distplot(train['Age'][train['Sex'] == 1], hist = False, ax = ax2, norm_hist = True, label = 'Female age distribution')

ax1.legend(loc = 'upper right')

ax2.legend(loc = 'upper right')
# We will now create age groups and check survival rate.

cut_offs = [0,15,30,80]

temp = pd.DataFrame(columns = ['Sex','Survived','age_group'])

for i in range(1,len(cut_offs)):

    df = train[["Survived",'Sex']][train['Age']>cut_offs[i-1]][train['Age']<=cut_offs[i]].groupby(['Sex'],as_index=False).mean()

    df['age_group'] = 'less than ' + str(cut_offs[i])

    temp = temp.append(df, ignore_index = True)
ax = sns.barplot(x = 'age_group', y = 'Survived', hue = 'Sex', data = temp)

ax.legend(bbox_to_anchor=(1.25, 1))
# Let us map values in age column to appropriate age groups.

train['Age'] = train['Age'].apply(lambda x: 1 if x <= 15 else 2 if x <= 30 else 3)

test['Age'] = test['Age'].apply(lambda x: 1 if x <= 15 else 2 if x <= 30 else 3)



train = pd.get_dummies(data = train, columns = ['Age'])

test = pd.get_dummies(data = test, columns = ['Age'])



# 2nd age group has lowest survival rate overall, so we will treat that as a base case and delete that column.

train.drop(['Age_2'], axis = 1, inplace = True)

test.drop(['Age_2'], axis = 1, inplace = True)
# 'Fare' is expected to correlate with 'Pclass'

f, [ax1, ax2] = plt.subplots(1,2, figsize = (20,5))

sns.barplot(hue = 'Embarked', y = 'Fare', x = 'Pclass', data = train[train['Cabin'] == 0], ax = ax1, hue_order = ['S','C','Q'])

sns.barplot(hue = 'Embarked', y = 'Fare', x = 'Pclass', data = train[train['Cabin'] == 1], ax = ax2, hue_order = ['S','C','Q'])

ax1.set_title('Doesn\'t have Cabin')

ax2.set_title('Has Cabin')

ax2.set_ylim([0,180])

plt.show()
def map_fare(x, cut_offs = None):

    if cut_offs == None:

        cut_offs = train['Fare'].describe()[['min','25%','50%','75%','max']]

    cut_offs = np.sort(cut_offs)

    for i in range(1,len(cut_offs)):

        if x <= cut_offs[i]:

            return i
# Let us find Pearson correlation coefficient between Pclass and mapped values of 'Fare'

mapped_fares = train['Fare'].apply(map_fare)

mapped_fares.corr(train['Pclass'])
test['Fare'] = test['Fare'].apply(map_fare)

train['Fare'] = mapped_fares
train = pd.get_dummies(train, columns = ['Embarked', 'title'])

test = pd.get_dummies(test, columns = ['Embarked', 'title'])



train.drop(['Embarked_S','title_rare'], inplace = True, axis = 1)

test.drop(['Embarked_S','title_rare'], inplace = True, axis = 1)
train.drop('PassengerId', axis = 1, inplace = True)

test_passenger_id = test['PassengerId']

test.drop('PassengerId', axis = 1, inplace = True)
# Let us check dtypes to ensure every variable is numeric.

train.dtypes
test.dtypes
correlations = train.corr()
d = {}

for col in correlations:

    temp = correlations[col].drop(col)

    for row in temp.index:

        if abs(temp[row]) > 0.5:

            if row + ' - ' + col not in d:

                d[col + ' - ' + row] = float('{:.4f}'.format(temp[row]))

d
# If you wish to see diagramatic representation of correlations, you can 'uncomment' code in this cell.



# f, ax = plt.subplots(1,1,figsize = (12,12))

# sns.set(font_scale = 1)

# sns.heatmap(correlations,square=True, annot=True, ax = ax, cmap = 'PuBu', cbar=True,

#             cbar_kws={"shrink": 0.75}, fmt = '.2f')

# plt.setp(ax.get_xticklabels(), fontsize=14)

# plt.setp(ax.get_yticklabels(), fontsize=14)

# plt.show()
seed = 0  # Seed to use when calling functions involving random selection. Important for reproducibility

kf = KFold(n_splits = 4, random_state = seed)

survived = train['Survived']

train.drop('Survived', axis = 1, inplace = True)
list_of_indices = []

for (_, temp) in kf.split(train.index):

    for index in temp:

        list_of_indices.append(index)

train_predictions = pd.DataFrame(index = list_of_indices)

test_predictions = pd.DataFrame()
def train_model(clf_name, clf, prediction_df, train_df, test_df):

    prediction_df[clf_name] = [-1]*prediction_df.shape[0]

    temp = pd.DataFrame()

    

    for i, (train_index, test_index) in enumerate(kf.split(train_df.index)):

        x = train_df.loc[train_index]

        y = survived.loc[train_index]

        test_values = train_df.loc[test_index]

        

        clf.fit(x,y)

        

        prediction_df[clf_name].loc[test_index] = list(clf.predict(test_values))

        temp[i] = list(clf.predict(test_df))

        

    test_predictions[clf_name] = temp.apply(lambda x: x.value_counts().index[0], axis = 1)
# Initialize the model with desired parameters.

lr = LogisticRegression(random_state = seed)

train_model(clf_name = 'logistic_regression', clf = lr, prediction_df = train_predictions, train_df = train, test_df = test)
# Initialize the model with desired parameters.

svc = SVC(random_state = seed, kernel = 'linear', C = 0.025)

train_model(clf_name = 'SVC', clf = svc, prediction_df = train_predictions, train_df = train, test_df = test)
# Initialize the model with desired parameters.

dtc = DecisionTreeClassifier(random_state = seed, max_depth = 10, min_samples_split = 30)

train_model(clf_name = 'decision_tree_classifier', clf = dtc, prediction_df = train_predictions, train_df = train, test_df = test)
# Initialize the model with desired parameters.

rfc = RandomForestClassifier(random_state = seed, n_estimators = 500, warm_start = True,

                             max_depth = 5, min_samples_leaf = 5)

train_model(clf_name = 'random_forest_classifier', clf = rfc, prediction_df = train_predictions, train_df = train, test_df = test)
# Initialize the model with desired parameters.

etc = ExtraTreesClassifier(random_state = seed, n_estimators = 500, warm_start = True,

                             max_depth = 8, min_samples_leaf = 5)

train_model(clf_name = 'extra_trees_classifier', clf = etc, prediction_df = train_predictions, train_df = train, test_df = test)
# Initialize the model with desired parameters.

gbc = GradientBoostingClassifier(random_state = seed, n_estimators = 50, warm_start = True, learning_rate = 0.1,

                                 max_depth = 5, min_samples_leaf = 25)

train_model(clf_name = 'gradient_boosting_classifier', clf = gbc, prediction_df = train_predictions, train_df = train, test_df = test)
# Initialize the model with desired parameters.

abc = AdaBoostClassifier(random_state = seed, n_estimators = 500)

train_model(clf_name = 'ada_boost_classifier', clf = abc, prediction_df = train_predictions, train_df = train, test_df = test)
# Initialize the model with desired parameters.

knc = KNeighborsClassifier(p = 2, n_neighbors = 3)

train_model(clf_name = 'k_neighbors_classifier', clf = knc, prediction_df = train_predictions, train_df = train, test_df = test)
accuracy = {}

for col in train_predictions.columns:

    accuracy[col] = sum([1 if train_predictions[col].loc[i] == survived.loc[i] else 0 for i in survived.index])/791
fig, ax = plt.subplots(1,1, figsize = (10,5))

sns.barplot(x = sorted(accuracy, key = accuracy.get, reverse = True), y = np.sort(list(accuracy.values()))[::-1],

            ax = ax, color = 'c')

for label in ax.get_xticklabels():

    label.set_rotation(90)

    label.set_fontsize(15)
auc_score = {}

for col in train_predictions.columns:

    auc_score[col] = roc_auc_score(survived, train_predictions[col])
fig, ax = plt.subplots(1,1, figsize = (10,5))

sns.barplot(x = sorted(auc_score, key = auc_score.get, reverse = True), y = np.sort(list(auc_score.values()))[::-1],

            ax = ax, color = 'c')

for label in ax.get_xticklabels():

    label.set_rotation(90)

    label.set_fontsize(15)
corr = train_predictions.corr()

f, ax = plt.subplots(1,1,figsize = (12,12))

sns.set(font_scale = 1)

sns.heatmap(corr,square=True, annot=True, ax = ax, cmap = 'PuBu', cbar=True,

            cbar_kws={"shrink": 0.75}, fmt = '.2f')

plt.setp(ax.get_xticklabels(), fontsize=14)

plt.setp(ax.get_yticklabels(), fontsize=14)

plt.show()
first_level_models = train_predictions.columns
train_predictions['majority_voting_all_models'] = train_predictions[first_level_models].apply(lambda x: x.value_counts().index[0], axis = 1)

test_predictions['majority_voting_all_models'] = test_predictions[first_level_models].apply(lambda x: x.value_counts().index[0], axis = 1)

accuracy['majority_voting_all_models'] = sum([1 if train_predictions['majority_voting_all_models'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791

accuracy['majority_voting_all_models']
# 'logistic_regression','extra_trees_classifier','gradient_boosting_classifier' and 'random_forest_classifier' are top 4 models.

# But 'extra_trees_classifier' and 'random_forest_classifier' have high correlation, so we'' choose only 1 out of these 2.

selected_cols = ['logistic_regression','extra_trees_classifier','gradient_boosting_classifier']

train_predictions['majority_voting_selected_cols'] = train_predictions[selected_cols].apply(lambda x: x.value_counts().index[0],

                                                                                            axis = 1)

test_predictions['majority_voting_selected_cols'] = test_predictions[selected_cols].apply(lambda x: x.value_counts().index[0],

                                                                                            axis = 1)

accuracy['majority_voting_selected_cols'] = sum([1 if train_predictions['majority_voting_selected_cols'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791

accuracy['majority_voting_selected_cols']
lr_second_level = LogisticRegression(random_state = seed)

train_model(clf_name = 'logistic_regression_second_level', clf = lr_second_level, prediction_df = train_predictions,

            train_df = train_predictions[first_level_models], test_df = test_predictions[first_level_models])

accuracy['logistic_regression_second_level'] = sum([1 if train_predictions['logistic_regression_second_level'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791

accuracy['logistic_regression_second_level']
lr_second_level_selected_cols = LogisticRegression(random_state = seed)

train_model(clf_name = 'logistic_regression_second_level_selected_cols', clf = lr_second_level, prediction_df = train_predictions,

            train_df = train_predictions[selected_cols], test_df = test_predictions[selected_cols])

accuracy['logistic_regression_second_level_selected_cols'] = sum([1 if train_predictions['logistic_regression_second_level_selected_cols'].loc[i] == survived.loc[i] else 0 for i in survived.index])/791

accuracy['logistic_regression_second_level_selected_cols']
most_accurate_clfs = sorted(accuracy, key = accuracy.get, reverse = True)

most_accurate_clfs
submission = pd.DataFrame({'PassengerId' : test_passenger_id,

                          'survived' : test_predictions['logistic_regression_second_level_selected_cols']})

submission.to_csv('titanic.csv', index = False)
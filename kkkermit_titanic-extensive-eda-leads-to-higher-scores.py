import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from IPython.display import Image, display
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import re
import math
# read the data in 
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
combined = train.append(test, sort=False)

print('Training set:')
print('Number of samples: {}'.format(train.shape[0]))
print('Number of variables: {}'.format(train.shape[1]))
print('Variables: {}\n'.format(train.columns.tolist()))
print('Testing set:')
print('Number of samples: {}'.format(test.shape[0]))
print('Number of variables: {}'.format(test.shape[1]))
print('Variables: {}\n'.format(test.columns.tolist()))
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train.head(5)
display(Image('https://upload.wikimedia.org/wikipedia/commons/5/5d/Titanic_side_plan_annotated_English.png'))
combined.nunique()
combined.Cabin.value_counts().head()
combined.loc[(combined.Parch == 0) & (combined.SibSp == 0), 'Ticket'].value_counts().head()
combined.Name.value_counts()[combined.Name.value_counts() > 1]
combined.loc[(combined.Name == 'Connolly, Miss. Kate') | (combined.Name == 'Kelly, Mr. James'),:].sort_values('Name')
# extract prefix from Name
combined['prefix'] = combined.Name.str.extract(r'(\, .*?\.)')
combined['prefix'] = combined.prefix.str.replace('\, ', '')

fem_prefix = ['Miss.', 'Mrs.', 'Mme.', 'Mlle.']
male_prefix = ['Mr.', 'Mister.', 'Master.']

inconsistent_index = []
for i, val in combined.iterrows():
    if val['prefix'] in fem_prefix:
        if val['Sex'] == 'Male':
            inconsistent_index.append(i)
    elif val['prefix'] in male_prefix:
        if val['Sex'] == 'Female':
            inconsistent_index.append(i)

print('Number of Prefix and Sex inconsistency: {}'.format(len(inconsistent_index)))
combined[(combined.Parch > 2) & (combined.Age <= 17)]
combined.loc[(combined.Parch > 2) & (combined.Age <= 17), ['SibSp', 'Parch']] = [3, 1]
Fords = ['Ford, Mr. William Neal', 'Ford, Miss. Robina Maggie "Ruby"', 'Ford, Miss. Doolina Margaret "Daisy"', 'Ford, Mrs. Edward (Margaret Ann Watson)', 'Ford, Mr. Edward Watson']

combined.loc[combined.Name.isin(Fords)]
sibs = ['Ford, Miss. Robina Maggie "Ruby"', 'Ford, Miss. Doolina Margaret "Daisy"', 'Ford, Mr. Edward Watson']

combined.loc[combined.Name.isin(sibs), ['SibSp', 'Parch']] = [3, 1]
combined.Age.describe()
combined.Fare.describe()
combined[combined.Fare == 0]
combined[['Survived', 'Pclass', 'Name', 'Fare', 'Cabin', 'Ticket']].sort_values('Fare', ascending = False).head(20)
# extract prefix from Name
combined['prefix'] = combined.Name.str.extract(r'(\, .*?\.)')
combined['prefix'] = combined.prefix.str.replace('\, ', '')

combined.prefix.value_counts()
prefix_dict = {
    'Capt.' : 'Officer',
    'Col.' : 'Officer',
    'Don.' : 'Royalty',
    'Dona.' : 'Royalty',
    'Dr.' : 'Officer',
    'Jonkheer.' : 'Royalty',
    'Lady.' : 'Royalty',
    'Major.' : 'Officer',
    'Master.' : 'Master',
    'Miss.' : 'Miss',
    'Mlle.' : 'Miss',
    'Mme.' : 'Miss',
    'Mr.' : 'Mr',
    'Mrs.' : 'Mrs',
    'Ms.' : 'Mrs',
    'Rev.' : 'Officer',
    'Sir.' : 'Royalty',
    'the Countess.' : 'Royalty'
}

combined.prefix = combined.prefix.apply(lambda row: prefix_dict[row])
combined.prefix.value_counts()
# Deck of cabin
# 'M' represents missing
combined.Cabin = combined.Cabin.apply(lambda row: row[0] if pd.notnull(row) else 'M')

combined.Cabin.value_counts()
combined['logFare'] = np.log10(combined.Fare + 1)
combined['FamilySize'] = combined.Parch + combined.SibSp
combined['noFamily'] = combined.apply(lambda row: 1 if row['FamilySize'] == 0 else 0, axis=1)
# split into train/test with new variable
train = combined.iloc[:891,:]
test = combined.iloc[891:,:]
tickets = set()
for ticket in train.Ticket:
    tickets.add(ticket)

ind = 0
for ticket in test.Ticket:
    if ticket in tickets:
        ind += 1
    else:
        tickets.add(ticket)

nonfam_tickets = set()
ind_2 = 0
for ticket in train.loc[train.noFamily == 1, 'Ticket']:
    nonfam_tickets.add(ticket)

for ticket in test.loc[test.noFamily == 1, 'Ticket']:
    if ticket in nonfam_tickets:
        ind_2 += 1
    else:
        nonfam_tickets.add(ticket)
        
print('{} group tickets out of {} tickets show up in both training and testing set, \n among them, {} were from families and {} were not.'.format(ind, len(tickets), ind - ind_2, ind_2))
print('Survival rate in training set: {}'.format(round(train.Survived.mean(), 2)))
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
sns.countplot(train.Survived)
# 'M' represents missing
combined.Cabin = combined.Cabin.apply(lambda row: row[0] if pd.notnull(row) else 'M')

# T only consists of 1 passenger and is from class1
# replace cabin T with C cause C has the most people from class 1
combined.loc[combined.Cabin == 'T', 'Cabin'] = 'C'
cate_variables = ['Sex', 'Embarked', 'Pclass', 'Parch', 'SibSp', 'FamilySize', 'Cabin']

def cate_countplot(data, var_list, cols=4, width=16, height=8, hspace=0.3, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    sns.set(font_scale=1.5)
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(len(var_list) / cols)
    
    for i, var in enumerate(var_list):
        ax = fig.add_subplot(rows, cols, i+1)
        sns.countplot(var, data=data)
        plt.xlabel(var, weight='bold')
        
cate_countplot(data=combined, var_list=cate_variables)
def create_bars(data, variables, cols=4, width=16, height=8, hspace=0.3, wspace=0.5):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(len(variables) / cols)

    for i, column in enumerate(data[variables].columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.barplot(column,'Survived', data=data.sort_values(column))
        plt.xticks(rotation=0)
        plt.xlabel(column, weight='bold')

create_bars(combined, cate_variables)
# survival rate on different features
# numerical variables
num_variables = ['Age', 'Fare', 'logFare']

# you are encouraged to play around with the bins and see the effect on plots
bins = [range(0, 81, 1), range(0, 300, 4), np.arange(0, 1.7, 0.02)]
def create_hists(data, variables, bins = bins, cols=3, width=20, height=6, hspace=0.5, wspace=0.5):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(variables) / cols)
    
    survived = data[data['Survived'] == 1]
    passed = data[data['Survived'] == 0]

    for i, column in enumerate(data[variables].columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.distplot(survived[column].dropna(), bins = bins[i], kde = False, color = 'blue')
        sns.distplot(passed[column].dropna(), bins = bins[i], kde = False, color = 'red')
        plt.xticks(rotation=0)
        plt.xlabel(column, weight='bold')
        plt.legend(['Survived', 'Deceased'])
        
create_hists(combined, num_variables)
variables_list = ['Survived', 'Age', 'Fare', 'logFare', 'Parch', 'SibSp', 'FamilySize']

plt.figure(figsize=(10, 10))
sns.heatmap(combined.iloc[:891,][variables_list].corr(), annot=True, fmt='.2f', square=True)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(14, 4))
fig.subplots_adjust(wspace=0.3, hspace=0.5)
ax = fig.add_subplot(1, 2, 1)
sns.countplot(combined.Pclass, hue=combined.Sex)
plt.title('Sex Count by Pclass ')

ax = fig.add_subplot(1, 2, 2)
sns.barplot(x=train.Pclass, y=train.Survived, hue=train.Sex)
plt.legend(title = 'Sex', loc='upper right')
t = plt.title('Survival Rate by Pclass & Sex')
class_list = [1, 2, 3]
def age_dists_by_class(data, class_list, cols=3, width=16, height=6, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(class_list) / cols)
    
    colors = ['blue', 'green', 'pink']
    bins = range(0, 80, 5)
    for i, cls in enumerate(class_list):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.distplot(data[(data.Age.notnull()) & (data.Pclass == cls)].Age, bins=bins, color=colors[i])
        plt.axvline(data[data.Pclass == cls].Age.mean(), color = 'red', label='Mean')
        plt.axvline(data[data.Pclass == cls].Age.median(), color = 'blue', label='Median')
        plt.legend(loc='upper right')
        plt.title('Class {}'.format(str(cls)))

age_dists_by_class(combined, class_list)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(6, 4))
sns.kdeplot(combined.loc[combined.Sex == 'male', 'Age'].dropna(), color='blue', label = 'male',shade=True)
sns.kdeplot(combined.loc[combined.Sex == 'female', 'Age'].dropna(), color='red', label = 'female', shade=True)
plt.xlabel('Age', weight='bold')
t = plt.title('Age by Sex')
# sex_age_vs_class
sex_list = ['male', 'female']
def sex_age_by_class(data, sex_list, cols=3, width=18, height=10, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(sex_list) * 2 / cols)
    
    survived = data[data.Survived == 1]
    passed = data[data.Survived == 0]
    
    ind = 1
    for i, sex in enumerate(sex_list):
        for j, cls in enumerate([1, 2, 3]):
            ax = fig.add_subplot(rows, cols, ind)
            sns.kdeplot(survived.loc[(survived.Pclass == cls) & (survived.Sex == sex), 'Age'].dropna(), color='blue', shade=True)
            sns.kdeplot(passed.loc[(passed.Pclass == cls) & (passed.Sex == sex), 'Age'].dropna(), color='red', shade=True)
            plt.legend(['Survived', 'Passed'], loc='upper right')
            plt.xlabel('Age', weight='bold')
            plt.title('Class {} - {}'.format(str(cls), sex))
            ind += 1
        
sex_age_by_class(combined, sex_list)
sex_list = ['female', 'male']
def fare_dists_by_sex(data, sex_list, cols=3, width=16, height=4, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(sex_list) / cols)
    
    colors = ['blue', 'pink']
    bins = range(0, 400, 40)
    for i, sex in enumerate(sex_list):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.distplot(data[(data.Fare.notnull()) & (data.Sex == sex)].Fare, bins=bins, color = colors[i])
#         plt.axvline(data[data.Sex == sex].Fare.mean(), color = 'red', label='Mean')
#         plt.axvline(data[data.Sex == sex].Fare.median(), color = 'blue', label='Median')
#         plt.legend(loc='upper right')
        plt.title(str(sex))

fare_dists_by_sex(combined, sex_list)
class_list = [1, 2, 3]
def fare_dists_by_class(data, class_list, cols=3, width=16, height=4, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(class_list) / cols)
    
    colors = ['blue', 'green', 'pink']
    
    for i, cls in enumerate(class_list):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.distplot(data[(data.Fare.notnull()) & (data.Pclass == cls)].Fare, color = colors[i])
        plt.axvline(data[data.Pclass == cls].Fare.mean(), color = 'red', label='Mean')
        plt.axvline(data[data.Pclass == cls].Fare.median(), color = 'blue', label='Median')
        plt.legend(loc='upper right')
        plt.title('Class {}'.format(str(cls)))

fare_dists_by_class(combined, class_list)
class_list = [1, 2, 3]
def class_violins(data, class_list, cols=3, width=16, height=6, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(class_list) / cols)

    for i, cls in enumerate(class_list):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.violinplot(x='Survived', y='Fare', data=combined[combined.Pclass == cls], palette='Set1')
        plt.title('Class {}'.format(str(cls)))
        ax.set_xticklabels(['Deceased','Survived'])
        plt.xlabel('')

class_violins(combined, class_list)
class_list = [1, 2, 3]
def class_violins_by_sex(data, class_list, cols=3, width=16, height=8, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(class_list) / cols)

    for i, cls in enumerate(class_list):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax = sns.violinplot(x='Survived', y='Fare', hue='Sex', data=combined[combined.Pclass == cls].sort_values('Sex'))
        plt.legend(loc='upper left')
        ax.set_xticklabels(['Deceased','Survived'])
        plt.title('Class {}'.format(str(cls)))
        plt.xlabel('')

class_violins_by_sex(combined, class_list)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(6, 4))
table = pd.crosstab(combined.Embarked, combined.Pclass)
table = pd.crosstab(combined.Embarked, combined.Pclass)
table = table.div(table.sum(axis=1), axis=0)
table.plot(kind="bar", stacked=True)
plt.xlabel('Embarked')
plt.ylabel('Fraction of Total')
plt.legend(title='Pclass', loc='lower right', bbox_to_anchor=(1.25, 0))
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(6, 4))
table = pd.crosstab(combined.Embarked, combined.Pclass)
sns.countplot(combined.Embarked, hue=combined.Sex)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(6, 4))
sns.barplot(x='Cabin', y='Survived', data=combined.sort_values(by=['Cabin']))
#cabin by class

table = pd.crosstab(combined.Cabin, combined.Pclass)
print(table)
table = table.div(table.sum(axis=1), axis=0)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(6, 4))
table.plot(kind='bar', stacked=True)
plt.ylabel('Fraction of Total')
plt.legend(title='Pclass', loc='lower right', bbox_to_anchor=(1.25, 0))
# cabin by sex
table = pd.crosstab(combined.Cabin, combined.Sex)
print(table)
table = table.div(table.sum(axis=1), axis=0)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(6, 4))
table.plot(kind='bar', stacked=True)
plt.xticks(rotation=0)
plt.legend(loc='lower right', bbox_to_anchor=(1.4, 0))
table = pd.crosstab(combined.Pclass, combined.FamilySize)
print(table)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
sns.pointplot(x='FamilySize', y='Survived', hue='Pclass', data=combined)
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
sns.pointplot(x='FamilySize', y='Survived', hue='Sex', data=combined)
# sex_age_vs_class
def survival_by_class_sex(data, cols=3, width=18, height=4, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = 2

    for i, cls in enumerate([1, 2, 3]):
        ax = fig.add_subplot(1, 3, i+1)
        sns.pointplot(x='FamilySize', y='Survived', hue='Sex', data=data[data.Pclass == cls].sort_values('Sex'))
        if i == 2:
            plt.legend(loc='upper right')
        else:
            plt.legend(loc='lower right')
        plt.title('Class {}'.format(str(cls)))
        
survival_by_class_sex(combined, sex_list)
combined[combined.Fare.isnull()]
F_fill = train[(train.Pclass == 3) & (train.noFamily == 1)].Fare.mean()
combined.loc[combined.Fare.isnull(), 'Fare'] = F_fill
combined.loc[combined.logFare.isnull(), 'logFare'] = np.log(F_fill)
combined[combined.Embarked.isnull()]
combined.loc[combined.Embarked.isnull(), 'Embarked'] = train[train.Pclass == 3].Embarked.mode()
prefix_list = ['Master', 'Miss', 'Mr', 'Mrs', 'Officer', 'Royalty']
def hists_by_prefix(data, prefix_list, cols=3, width=15, height=8, hspace=0.5, wspace=0.25):
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    rows = math.ceil(len(prefix_list) / cols)
    
    colors = ['blue', 'green', 'yellow', 'pink', 'orange', 'grey']
    
    for i, prefix in enumerate(prefix_list):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.distplot(data[(data.Age.notnull()) & (data.prefix == prefix)].Age, color = colors[i])
        plt.axvline(data[data.prefix == prefix].Age.mean(), color = 'blue')
        plt.axvline(data[data.prefix == prefix].Age.median(), color = 'red')
        plt.title(prefix)

hists_by_prefix(combined, prefix_list)
# dict of mean for each prefix group
prefix_mean = train.groupby('prefix').Age.mean().to_dict()

# impute missing age
combined.Age = combined.groupby('prefix').Age.transform(lambda x: x.fillna(x.mean()))
# get_dummies
combined_cate_var = pd.get_dummies(combined[['Cabin', 'Embarked', 'prefix']])

# include Pclass
combined_cate_var['Pclass'] = combined.Pclass
combined_cate_var['noFamily'] = combined.noFamily
combined_cate_var['Sex'] = combined.Sex.map({'male':1, 'female':0})

combined_cate_var.columns
num_list = ['Age', 'SibSp', 'Parch', 'logFare', 'FamilySize']

combined_num = combined[num_list].copy()

combined_num[num_list] = StandardScaler().fit_transform(combined_num[num_list].values)
# combined features
combined_processed = pd.concat([combined_cate_var, combined_num], axis=1)

combined_processed.head(5)
# train test split 
x_train = combined_processed.iloc[:891,:]
y_train = combined.iloc[:891,:].Survived

x_test = combined_processed.iloc[891:,:]
# define model
baseline_model = RandomForestClassifier(random_state=4, n_estimators=100, min_samples_split=15, oob_score=True)

# fit
baseline_model.fit(x_train, y_train)

# feature importance
feature_importances = pd.DataFrame({'Importance Score': baseline_model.feature_importances_}, index=x_train.columns).sort_values(by='Importance Score', ascending=False)

# top 10 features by importance
feature_importances[:10]
rfe = RFE(estimator=baseline_model, n_features_to_select=1, step=1)
rfe.fit(x_train, y_train)
importance = pd.DataFrame()
importance['features'] = x_train.columns.values
importance['ranking'] = rfe.ranking_
importance = importance.sort_values('ranking')

cv_result = pd.DataFrame()
cv_result['num_features'] = range(1, 11, 1)
cv_result['training_score'] = 0
cv_result['testing_score'] = 0
cv_result['testing_std'] = 0

for i in range(1, 11, 1):
    # top features
    features = importance['features'].iloc[:i]
    
    # compute training and CV accuracy
    cv_score = cross_validate(baseline_model, x_train[features], y_train, cv=10, return_train_score=True)
    cv_result.loc[i-1, 'training_score'] = cv_score['train_score'].mean()
    cv_result.loc[i-1, 'testing_score'] = cv_score['test_score'].mean()
    cv_result.loc[i-1, 'testing_std'] = cv_score['test_score'].std()
sns.set(font_scale=1.5)
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(8, 6))
sns.lineplot(cv_result['num_features'], cv_result['training_score'], marker="o", label='Training')
sns.lineplot(cv_result['num_features'], cv_result['testing_score'], marker="^", label='Testing')
t = plt.xticks(range(0, 11, 1))
plt.xlabel('Number of Features')
plt.ylabel('CV Accuracy')
plt.legend()
importance['features'].iloc[:5]
tree_param_grid = {'max_depth': [10, 20, 30, 40],
 'min_samples_split': [2, 5, 10, 15, 20, 25],
 'n_estimators': [50, 100, 150, 200]}

grid = GridSearchCV(RandomForestClassifier(), param_grid=tree_param_grid, cv=10, n_jobs=-1)
grid.fit(x_train[importance['features'].iloc[:5]], y_train)
print('Best CV score {} was achieved with:'.format(round(grid.best_score_, 4)))
params = grid.best_estimator_.get_params()
print('max_depth: {}'.format(str(params['max_depth'])))
print('min_samples_split: {}'.format(str(params['min_samples_split'])))
print('n_estimators: {}'.format(str(params['n_estimators'])))
model = RandomForestClassifier(**params)
model.fit(x_train[importance['features'].iloc[:6]], y_train)
prediction = model.predict(x_test[importance['features'].iloc[:6]]).astype(int)
submission = pd.DataFrame(test.PassengerId.copy())
submission['Survived'] = pd.Series(prediction)
submission.to_csv('submission.csv', index = False)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# get mean of passenger's age from training set
age_mean = train.Age.mean()

# replace in training set
train.Age = train.apply(lambda row: age_mean if pd.isnull(row['Age']) else row['Age'], axis=1)

# same for testing set
test.Age = test.apply(lambda row: age_mean if pd.isnull(row['Age']) else row['Age'], axis=1)
# define the model
# same model that we use before
baseline_model = RandomForestClassifier(random_state=4, n_estimators=100, min_samples_split=15, oob_score=True)

# separate training and target variables
X = train[['Sex', 'Age', 'Pclass']].copy()
Y = train['Survived']

# encode Sex 
# 0 for female, 1 for male
X.Sex = X.apply(lambda row: 0 if row['Sex'] == 'female' else 1, axis=1)

# fit
baseline_model.fit(X, Y)

print('OOB score of baseline model: {}'.format(round(baseline_model.oob_score_, 3)))
# data processing on test set
test_X = test[['Sex', 'Age', 'Pclass']].copy()
test_X.Sex = test_X.apply(lambda row: 0 if row['Sex'] == 'female' else 1, axis=1)

# make prediction on test data
baseline_prediction = baseline_model.predict(test_X)

# produce csv file for submission
baseline_submission = pd.DataFrame(test.PassengerId.copy())
baseline_submission['Survived'] = pd.Series(baseline_prediction)
baseline_submission.to_csv('baseline_submission.csv', index = False)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combined = train.append(test, sort=False)
class_mean = train.groupby('Pclass').Age.mean().to_dict()

train.Age = train.groupby('Pclass').Age.transform(lambda x: x.fillna(x.mean()))
test.Age = test.apply(lambda row: class_mean[row['Pclass']], axis=1)

X = train[['Sex', 'Age', 'Pclass']].copy()
Y = train['Survived']

# encode Sex 
# 0 for female, 1 for male
X.Sex = X.apply(lambda row: 0 if row['Sex'] == 'female' else 1, axis=1)

# fit
baseline_model.fit(X, Y)

print('OOB score of baseline model: {}'.format(round(baseline_model.oob_score_, 3)))

# data processing on test set
test_X = test[['Sex', 'Age', 'Pclass']].copy()
test_X.Sex = test_X.apply(lambda row: 0 if row['Sex'] == 'female' else 1, axis=1)

# make prediction on test data
baseline_prediction = baseline_model.predict(test_X)

# produce csv file for submission
baseline_submission = pd.DataFrame(test.PassengerId.copy())
baseline_submission['Survived'] = pd.Series(baseline_prediction)
baseline_submission.to_csv('age_by_class_submission.csv', index = False)
baseline_model.fit(x_train[['Age', 'Sex', 'Pclass']], y_train)
prediction = baseline_model.predict(x_test[['Age', 'Sex', 'Pclass']]).astype(int)
submission = pd.DataFrame(test.PassengerId.copy())
submission['Survived'] = pd.Series(prediction)
submission.to_csv('age_by_prefix_submission.csv', index = False)
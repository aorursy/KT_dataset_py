import pandas as pd

import numpy as np

from IPython.display import display, Image, Javascript

from scipy.stats import scoreatpercentile, percentileofscore

import sys

import seaborn as sns

import matplotlib.pyplot as plt

%autosave 60

%matplotlib inline



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 20)

plt.rcParams['figure.figsize'] = [10, 5]
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve, accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.inspection import permutation_importance
submissions = pd.read_csv('../input/titanicpublicleaderboard/titanic-publicleaderboard.csv')

sub_max_scores = submissions.groupby('TeamId')['Score'].agg('max')

sub_max_scores = sub_max_scores[(sub_max_scores>0.4) & (sub_max_scores<1)]



score_to_beat = scoreatpercentile(sub_max_scores, 90)

print('The score to beat is: ', score_to_beat, '\n' + '-'*30)
ax = sns.distplot(sub_max_scores, label='scores distribution');

ax.axvline(score_to_beat, label='score to beat', linestyle='--', color='C1')

plt.legend();
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

CLASS_LABEL = 'Survived'

data = pd.concat([train, test], axis=0).reset_index(drop=True)

data_p = data.copy()
data_p.head()
(data_p.isna().sum().sort_values(ascending=False).replace(0, np.nan).dropna() / data_p.shape[0] * 100).to_frame('NaN %').T
print(*train.columns, sep='\n')
dummy_columns = []

dummy_columns.append('Pclass')
def get_title(name):

    import re

    match = re.search('\w{2,}\.\s', name)

    if match is None:

        return None

    else:

        return match.group()
def get_nickname(name):

    import re

    match = re.search('"[\w\s]{2,}"', name)

    if match is None:

        return None

    else:

        return match.group()
def get_pname(name):

    import re

    match = re.search('\([\w\s]{2,}\)', name)

    if match is None:

        return None

    else:

        return match.group()
data_p['Title'] = data['Name'].apply(get_title)

data_p['Nickname'] = data['Name'].apply(get_nickname)

data_p['Pname'] =  data['Name'].apply(get_pname)

data_p['CleanName'] = data_p.fillna('&&&').apply(lambda x: x['Name']

                                  .replace(x['Nickname'], ' ')

                                  .replace(x['Title'], ' ')

                                  .replace(x['Pname'], ' ')

                                  .replace('  ', ' ').replace('  ', ' ').replace('( )','').strip(', ')

                                  , axis=1)

data_p['CleanNameShort'] = data_p['CleanName'].apply(lambda x: ' '.join(x.split()[:2]))



data_p['Title'] = data_p['Title'].str.strip(' .')
UNMARRIED_TITLES = ['Mlle', 'Master', 'Miss']

data_p['Title_Unmarried'] = data_p['Title'].apply(lambda x: x in UNMARRIED_TITLES)
data_p['Title_small'] = data_p['Title'].replace(['Lady', 'Countess','Capt', 'Col',

                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 

                                                 'Jonkheer', 'Dona'], 'Rare')

data_p['Title_small'] = data_p['Title_small'].replace('Mlle', 'Miss')

data_p['Title_small'] = data_p['Title_small'].replace('Ms', 'Miss')

data_p['Title_small'] = data_p['Title_small'].replace('Mme', 'Mrs')

sns.countplot(data_p['Title_small']);
data_p['Sp_onboard'] = ((data_p.groupby(['CleanNameShort','Title_Unmarried'])['CleanNameShort'].transform('count')==2) \

                        &(data_p.groupby(['CleanNameShort','Title_Unmarried'])['Sex'].transform('nunique')>1)).astype(int)

data_p['Sib_onboard'] = data_p['SibSp'] - data_p['Sp_onboard']

data_p.loc[data_p['Sib_onboard']<0,'Sib_onboard'] = 0
MINIMAL_YEARS_FOR_PARENT = 13

def group_pa_pos(x):

    x_copy = x.copy()

    result = x.copy()

    for i, y in enumerate(x_copy):

        if pd.isna(y):

            result.iloc[i] = len(x_copy) - 1

        else:

            result.iloc[i] = (x_copy.fillna(80) > (y+MINIMAL_YEARS_FOR_PARENT)).sum()

    return result



def group_ch_pos(x):

    x_copy = x.copy()

    result = x.copy()

    for i, y in enumerate(x_copy):

        if pd.isna(y):

            result.iloc[i] = len(x_copy)-1

        else:

            result.iloc[i] = (x_copy.fillna(1) < (y-MINIMAL_YEARS_FOR_PARENT)).sum()

    return result



data_p['_ch_pos'] = data_p.groupby(['Ticket'])['Age'].transform(group_ch_pos)

data_p['Ch_onboard'] = data_p.apply(lambda x: min(x[['_ch_pos', 'Parch']]), axis=1)

data_p['Pa_onboard'] = data_p['Parch'] - data_p['Ch_onboard']

data_p['group_size'] = data_p.groupby(['Ticket'])['Ticket'].transform('count')

data_p['family_size'] = data_p['SibSp'] + data_p['Parch'] + 1

data_p['is_alone'] = (data_p['family_size'] == 1).astype(int)
data_p.loc[data_p['Ticket']=='19950', 

           ['Title', 'Name', 'Age', 'Sex', 'Parch', '_ch_pos', 'Pa_onboard', 'Ch_onboard', 'Sp_onboard', 'family_size']

          ].sort_values('Age', ascending=False)
OVERALL_SURVIVAL_RATE = data['Survived'].mean()

def group_survival_rate(x, na_value=OVERALL_SURVIVAL_RATE):

    x_copy = x.copy().fillna(na_value)

    result = x_copy.copy()

    people_in_group = result.count()

    if people_in_group == 1:

        return na_value

    for i, y in enumerate(x_copy):

        sum_survived_but_me = x_copy.sum() - y

        result.iloc[i] = sum_survived_but_me / (people_in_group-1.0)

    return result



data_p['group_survival_rate'] = data_p.groupby(['Ticket'])['Survived'].transform(group_survival_rate)
data_p['Sex_male'] = (data_p['Sex'] == 'male').astype(int)
print(data_p['Age'].isna().value_counts(True))
sns.distplot(data_p['Age']);
sns.countplot(data_p['SibSp']);
sns.countplot(data_p['Parch']);
# with help from pd_helpers.get_data_formata

ticket_formats = [

 '([A-Z]\\.){2,3} \\d{4,5}',

 'A[A-Z]?/\\d\\.? \\d{3,5}',

 'A\\.[/ ][25]\\. \\d{4,5}',

 'A\\.?[45]\\. \\d{4,5}',

 'CA\\. 23\\d{2}',

 'LINE',

 'SC/AH Basle 541',

 'SC/A\\.3 \\d{3,4}',

 'SC/Paris 21\\d3',

 'SO?TON/O[\\s\\.]?[2Q]\\. 3\\d{5,6}',

 'S[A-Z]{1,4}/[A-Z]\\d \\d{5,7}',

 'S\\.[CO]\\./[AP]\\.[4P]\\. \\d{1,5}',

 'W\\./C\\. \\d{4,5}',

 '[A-Z]\\.[A-Z]\\./[A-Z]{2,5} \\d{3,5}',

 '[A-Za-z]{1,2} \\d{4,6}',

 '[A-Z]{1,5}/[A-Z]{1,5} \\d{3,7}',

 '\\d{3,7}']
def replace_with_format(value, formats):

    if pd.isna(value):

        return None

    import re

    for f in formats:

        if re.match('^' + f + '$', value):

            return f

    return None



data_p['Ticket'] = data_p['Ticket'].apply(str)

data_p['Ticket_format'] = data_p['Ticket'].apply(lambda v: replace_with_format(v, ticket_formats))

print(len(data_p[pd.isna(data_p['Ticket_format'])]['Ticket']),'unrecognized formats')

data_p['Ticket_format'].value_counts()
display(data_p[pd.isna(data['Fare'])])

passenger_1044_probable_fare = data_p[data_p['Pclass'] == data_p.loc[data_p['PassengerId']==1044, 'Pclass'].iloc[0]]['Fare'].dropna().median()

data_p['Fare'] = data_p['Fare'].fillna(passenger_1044_probable_fare)
sns.distplot(data_p['Fare']);
cabin_formats = [

 '([A-Z]\\d{2} ){1,3}[A-Z]\\d{2}',

 'F [A-Z]\\d{2}',

 '[A-Z]',

 '[A-Z]\\d{1,3}']
data_p['Cabin_format'] = data_p['Cabin'].apply(lambda v: replace_with_format(v,  pd.Series(cabin_formats)))



print((data_p[pd.isna(data_p['Cabin_format'])]['Cabin']).count(),'unrecognized formats')

data_p['Cabin_format'].value_counts(True)
sns.countplot(data_p['Embarked'])

data_p['Embarked'] = data_p['Embarked'].fillna('S')

dummy_columns.append('Embarked')
dummy_columns += ['Cabin_format', 'Ticket_format', 'Title_small']

data_features = pd.get_dummies(data_p, columns=list(set(dummy_columns)), drop_first=False)



data_features.drop(columns=['Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin', '_ch_pos', 'Name', 

                            'CleanName', 'CleanNameShort', 'Title'], inplace=True)



data_features['has_Pname'] = pd.notna(data_features['Pname']).astype(int)

data_features['has_Nickname'] = pd.notna(data_features['Nickname']).astype(int)

data_features['Title_Unmarried'] = data_features['Title_Unmarried'].astype(int)

data_features.drop(columns=['Pname', 'Nickname'], inplace=True)



data_features.set_index('PassengerId', inplace=True)
data_features.head()
for col in data_features.columns:

    if data_features[col].nunique() != 2:

        continue

    if data_features[col].value_counts(True)[0] > 0.99:

        print(f'Dropping column: {col}')

        data_features.drop(columns=col, inplace=True)
all_cols_but_nans = [v for v in data_features.columns if v not in ('Age', 'group_survival_rate', 'Survived')]

age_exists = pd.notna(data_features['Age'])

rfModel_age = RandomForestRegressor()

rfModel_age.fit(data_features[age_exists][all_cols_but_nans], data_features[age_exists]['Age'])



generatedAgeValues = rfModel_age.predict(X = data_features[~age_exists][all_cols_but_nans])

data_features.loc[~age_exists, 'Age'] = generatedAgeValues
sns.distplot(data_features['Age']);
# data_features.to_pickle('train_and_test_features.pkl')
# data_features = pd.read_pickle('train_and_test_features.pkl')

gender_submission = pd.read_csv('../input/titanic/gender_submission.csv').set_index('PassengerId')



# dropping columns which were found to have low feature importance (see bellow)

cols_to_drop = ['Ticket_format_([A-Z]\\.){2,3} \\d{4,5}',

 'Ticket_format_A[A-Z]?/\\d\\.? \\d{3,5}',

 'Ticket_format_SO?TON/O[\\s\\.]?[2Q]\\. 3\\d{5,6}',

 'Ticket_format_W\\./C\\. \\d{4,5}',

 'Ticket_format_[A-Z]{1,5}/[A-Z]{1,5} \\d{3,7}',

 'Ticket_format_[A-Za-z]{1,2} \\d{4,6}','Embarked_Q', 

 'Cabin_format_([A-Z]\\d{2} ){1,3}[A-Z]\\d{2}', 'Pa_onboard', 'Sp_onboard', 'Title_small_Rare', 

 'Ch_onboard',  'Ticket_format_\\d{3,7}', 'Embarked_S']

data_features.drop(columns = (cols_to_drop), inplace=True)

data_features['Pclass'] = data_features['Pclass_3'] * 3 + data_features['Pclass_2'] * 2 + data_features['Pclass_1']

data_features.drop(columns = ['Pclass_1', 'Pclass_2', 'Pclass_3'], inplace=True)

data_features['age*pclass'] = data_features['Age'] * data_features['Pclass']



CLASS_LABEL = 'Survived'

RSEED = np.random.randint(100)



data_features['Age'] = data_features['Age'].fillna(data_features['Age'].mean())

data_features.drop(columns=['group_survival_rate'])

test_set = data_features[pd.isna(data_features[CLASS_LABEL])].copy()

train_set = data_features[pd.notna(data_features[CLASS_LABEL])].copy()



train_set[CLASS_LABEL] = train_set[CLASS_LABEL].astype(int)

labels = np.array(train_set.pop(CLASS_LABEL))

test_set.drop(columns=CLASS_LABEL, inplace=True)
# 20% examples in test data

train, cv, train_labels, cv_labels = train_test_split(train_set,

                                         labels, 

                                         stratify = labels,

                                         test_size = 0.2, 

                                         random_state = RSEED)
features = list(train.columns)
model = RandomForestClassifier(n_estimators=50, 

                               random_state=RSEED, 

                               max_features = 'sqrt', max_depth=10, bootstrap=True)

# Fit on training data

model.fit(train, train_labels)
# Training predictions (to demonstrate overfitting)

train_rf_predictions = model.predict(train)

train_rf_probs = model.predict_proba(train)[:, 1]



# Testing predictions (to determine performance)

rf_predictions = model.predict(cv)

rf_probs = model.predict_proba(cv)[:, 1]
def plot_model_roc(predictions, probs, train_predictions, train_probs, test_labels):  

    

    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    model_auc = np.round(roc_auc_score(test_labels, probs), 3)

    

    # Plot

    plt.figure(figsize = (10, 10))

    plt.plot([0, 1], [0, 1], 'b', label = 'baseline')

    plt.plot(model_fpr, model_tpr, 'r', label = 'model (AUC = {})'.format(model_auc))

    plt.legend();

    plt.xlabel('False Positive Rate'); 

    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');

#     print(f'AUC : {model_auc}')

#     print(confusion_matrix(test_labels, rf_predictions))

    plt.show();



plot_model_roc(rf_predictions, rf_probs, train_rf_predictions, train_rf_probs, cv_labels)
print(f'Random Forest CV accuracy : {accuracy_score(cv_labels, rf_predictions)}')

print(f'Gender CV accuracy : {accuracy_score(cv_labels, 1-cv.Sex_male)}')
feature_names = np.r_[train_set.columns]

tree_feature_importances = model.feature_importances_

sorted_idx_rf = tree_feature_importances.argsort()



y_ticks = np.arange(0, len(features))

fig, ax = plt.subplots(figsize = (10, 10))

# ax.figure(figsize = (16, 6))

ax.barh(y_ticks, tree_feature_importances[sorted_idx_rf])

ax.set_yticklabels(feature_names[sorted_idx_rf])

ax.set_yticks(y_ticks)

ax.set_title("Random Forest Feature Importances (MDI)")

fig.tight_layout()

plt.show()
permutation_imp_data = permutation_importance(model, cv, cv_labels, n_repeats=10,

                                random_state=RSEED)

sorted_idx_pid = permutation_imp_data.importances_mean.argsort()
fig, ax = plt.subplots(figsize = (10, 10))

ax.boxplot(permutation_imp_data.importances[sorted_idx_pid].T,

           vert=False, labels=cv.columns[sorted_idx_pid])

ax.set_title("Permutation Importances (test set)")

fig.tight_layout()

plt.show()
# Fit on training data

model.fit(train_set, labels)
# Training predictions (to determine overfitting)

train_rf_predictions = model.predict(train_set)

train_rf_probs = model.predict_proba(train_set)[:, 1]



# Testing predictions (to determine performance)

rf_predictions = model.predict(test_set)

rf_probs = model.predict_proba(test_set)[:, 1]
prediction_set = test_set.copy()

prediction_set['Survived'] = rf_predictions

prediction_set = prediction_set[['Survived']]

prediction_set.head()
confusion_matrix(prediction_set['Survived'], gender_submission['Survived'])
prediction_set.to_csv('submission.csv')
MY_BEST_SCORE = 0.81339

submissions = pd.read_csv('../input/titanicpublicleaderboard/titanic-publicleaderboard.csv')

sub_max_scores = submissions.groupby('TeamId')['Score'].agg('max')

sub_max_scores = sub_max_scores[(sub_max_scores>0.4) & (sub_max_scores<1)]



score_to_beat = scoreatpercentile(sub_max_scores, 90)

print('The score to beat is:', score_to_beat)

print('You beat the score!' if score_to_beat < MY_BEST_SCORE else 'You didn''t beat the score...')

print('Your percentile is:', np.round(percentileofscore(sub_max_scores, MY_BEST_SCORE),2),'%')
ax = sns.distplot(sub_max_scores, label='Score distribution');

ax.axvline(MY_BEST_SCORE, color='C1', label='My score')

ax.axvline(score_to_beat, color='grey', linestyle='--', label='Top 10%')

plt.legend()

plt.show();
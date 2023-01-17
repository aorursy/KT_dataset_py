import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from xgboost import XGBClassifier
from xgboost import plot_importance

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
train_df = pd.read_csv("../input/train.csv")
test_df  = pd.read_csv("../input/test.csv")

(len(train_df), len(test_df))
# Full dataset is needed for imputing missing values & also for pruning outliers

train_len = len(train_df)
titanic_df = pd.concat([train_df, test_df], axis=0, ignore_index=True, sort=True)
titanic_df.info()
# Impute "Embarked" missing values with the most common value 'S'

sns.countplot(x='Embarked', data=titanic_df)
titanic_df['Embarked'] = titanic_df['Embarked'].fillna(value='S')
# Extract Title from Name, store in column and plot barplot

import re

titanic_df['Title'] = titanic_df.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

sns.countplot(x='Title', data=titanic_df);
plt.xticks(rotation=45);
# Replace rare Title with corresponding common Title

titanic_df['Title'] = titanic_df['Title'].replace({'Mlle': 'Miss', 
                                                   'Major': 'Mr', 
                                                   'Col': 'Mr', 
                                                   'Sir': 'Mr', 
                                                   'Don': 'Mr', 
                                                   'Mme': 'Miss', 
                                                   'Jonkheer': 'Mr', 
                                                   'Lady': 'Mrs', 
                                                   'Capt': 'Mr', 
                                                   'Countess': 'Mrs', 
                                                   'Ms': 'Miss', 
                                                   'Dona': 'Mrs'})

sns.countplot(x='Title', data=titanic_df);
plt.xticks(rotation=45);
# Impute "Age" by median of Age of Name's Title group

titles = ['Dr', 'Master', 'Miss', 'Mr', 'Mrs', 'Rev']
for title in titles:
    age_to_impute = titanic_df.groupby('Title')['Age'].median()[titles.index(title)]
    titanic_df.loc[(titanic_df['Age'].isnull()) & (titanic_df['Title'] == title), 'Age'] = age_to_impute
titanic_df['Familial'] = (titanic_df['SibSp'] + titanic_df['Parch']) > 0
# Impute "Fare" missing value
# Fare seem to be highly correlated to Pclass & the missing observation's Pclass is 3

medianFare = titanic_df[titanic_df['Pclass'] == 3]['Fare'].median()
titanic_df['Fare'] = titanic_df['Fare'].fillna(value = medianFare)
# Categorize continuous variables (Age into 16, i.e., bin width is 80/16)

custom_bucket_array = np.linspace(0, 80, 17)
titanic_df['CatAge'] = pd.cut(titanic_df['Age'], custom_bucket_array)
labels, levels = pd.factorize(titanic_df['CatAge'])
titanic_df['CatAge'] = labels
custom_bucket_array
custom_bucket_array = np.linspace(0, 520, 53)
titanic_df['CatFare'] = pd.cut(titanic_df['Fare'], custom_bucket_array)
labels, levels = pd.factorize(titanic_df['CatFare'])
titanic_df['CatFare'] = labels
custom_bucket_array
titanic_df['SexBool'] = titanic_df['Sex'].map({'male': 0, 'female': 1})
titanic_df['EmbarkedInt'] = titanic_df['Embarked'].map({'S': 0, 'C': 1, 'Q':2})
titanic_df['TitleInt'] = titanic_df['Title'].map({'Mr':0, 'Mrs':1, 'Miss':2, 'Master':3, 'Rev':4, 'Dr':5})
# Get back the features engineered train_df & test_df

train_df = titanic_df.loc[titanic_df['PassengerId'] <= train_len]
test_df = titanic_df.loc[titanic_df['PassengerId'] > train_len].iloc[:, titanic_df.columns != 'Survived']

(len(train_df), len(test_df))
# Heatmap to show Pearson Correlation of bivariate permutations

plt.figure(figsize=(14,12))
foo = sns.heatmap(train_df.drop(['PassengerId', 'Name', 'Title', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'CatFare', 'Cabin', 'Embarked'],axis=1).corr(), vmax=0.6, square=True, annot=True)
fig, axs = plt.subplots(ncols=2, figsize=(15,5))
axs[0].set_title('female')
sns.countplot(x='Survived', hue='Pclass', data=titanic_df.loc[titanic_df['Sex'] == 'female'], ax=axs[0])
axs[1].set_title('male')
sns.countplot(x='Survived', hue='Pclass', data=titanic_df.loc[titanic_df['Sex'] == 'male'], ax=axs[1])
# The Puzzle
sns.countplot(x='Survived', hue='Sex', data=titanic_df)
fig, axs = plt.subplots(ncols=2, figsize=(15,5))
axs[0].set_title('female')
sns.countplot(x='CatAge', hue='Survived', data=train_df.loc[train_df['Sex'] == 'female'], ax=axs[0])
axs[1].set_title('male')
sns.countplot(x='CatAge', hue='Survived', data=train_df.loc[train_df['Sex'] == 'male'], ax=axs[1])
fig, axs = plt.subplots(ncols=2, figsize=(15,5))
axs[0].set_title('female')
sns.countplot(x='Familial', hue='Survived', data=train_df.loc[train_df['Sex'] == 'female'], ax=axs[0])
axs[1].set_title('male')
sns.countplot(x='Familial', hue='Survived', data=train_df.loc[train_df['Sex'] == 'male'], ax=axs[1])
# Select feature column names and target variable we are going to use for training
# Best score with ['Pclass', 'Fare', 'Sex_binary', 'AgeCategoryIndex', 'Alone']

Columns = ['SexBool', 'Pclass', 'Fare', 'CatAge', 'Familial', 'EmbarkedInt', 'TitleInt']
Label = 'Survived'

train_X = train_df.loc[:, train_df.columns != 'Survived']
train_y = train_df['Survived']
# Instantiate XGB classifier - its hyperparameters are tuned through SkLearn Grid Search below

model = XGBClassifier()
# Performing grid search for important hyperparameters of XGBoost
# It has been observed that non-default value of only n_estimators is useful
# Other hyerparameters default values are the best (learning_Rate as 0.1, max_depth as 3, alpha L1 regularizer as 0 & lambda L2 regularizer as 1)

both_scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score), 'Loss':'neg_log_loss'}
params = {
        'n_estimators': [100, 200, 500, 1000, 1500],
        'learning_rate': [0.05, 0.1, 0.2]
        #'max_depth':[3, 4, 5]
        }
clf = GridSearchCV(model, params, cv=5, scoring=both_scoring, refit='AUC', return_train_score=True)
clf.fit(train_X[Columns], train_y)
print((clf.best_score_, clf.best_params_))
print("="*30)

print("Grid scores on training data:")
means = clf.cv_results_['mean_test_AUC']
stds = clf.cv_results_['std_test_AUC']
log_losses = clf.cv_results_['std_test_Loss']

for mean, std, log_loss, params in zip(means, stds, log_losses, clf.cv_results_['params']):
    print("AUC Score: %0.3f (+/-%0.03f); Log Loss: %0.3f for %r" % (mean, std * 2, log_loss, params))
# If grid params permutes across multiple hyperparameters, then below plot would have many lines (n1*n2*n3..) & may look cluttered
# Observe the best AUC & Accuracy

results = clf.cv_results_

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=16)

plt.xlabel("n_estimators: no of boosted trees")
plt.ylabel("AUC Score")

ax = plt.gca()
ax.set_xlim(80, 1020)
ax.set_ylim(0.7, 1)

X_axis = np.array(results['param_n_estimators'].data, dtype=float)

for scorer, color in zip(sorted(both_scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid('off')
plt.show()
#Make predictions using the features (Columns) from test_df

predictions = clf.predict(test_df[Columns]).astype(int)

submission = pd.DataFrame({'PassengerId':test_df['PassengerId'], 'Survived':predictions})
# Fill submission csv file
filename = 'submit.csv'
submission.to_csv(filename,index=False)
### EDA.x Extreme Fare which could possibly be outlier
'''
# 4 passengers with Fare > 512.0 of which 1 are from test_df (passenger id 1235)
# All on same Ticket 'PC 17755' => hence pid 1235 can be predicted as SURVIVED

# 17 passengers with Fare < 1.0 of which 2 are from test_df (passenger id 1158 on Ticket_112051 & 1264 on Ticket_112058)
# Both these passengers can be predicted as DIED

# TODO: Manual row append to 'submission' dataframe needs to be fixed
#titanic_df = titanic_df.loc[(titanic_df['Fare'] > 1.0) & (titanic_df['Fare'] < 512.0)]
#titanic_df = titanic_df.loc[titanic_df['Fare'] < 512.0]

# manual row append needs to be fixed... If we prune Fare > 512.0 which consists of 4 observations (3 train & 1 test), then below prediction must be manually added
#submission = submission.append({1235: 1}, ignore_index=True)
#sideEntryPrediction = [1235, 1]
#submission.loc[len(submission)] = sideEntryPrediction
#submission = submission.astype(int)
#submission.sort_values(by=['PassengerId','Survived'], ascending=True,inplace=True)
'''
### EDA.x: Though not useful... Survival trend among passengers on unique Ticket -vs- common/group Ticket appeared to be quite visible, hence 
# tried to split full dataset into Grouped & Single
#
# test/test_df (97 on group ticket -vs- 321 on single ticket)
# train_df (344 on group ticket -vs- 547 on single ticket)

# train_df has 344 passengers on Group Ticket
#Survived  0.0  1.0
#Sex               
#female     47  133
#male      118   46

# trainf_df has 547 passengers with Single Ticket
#Survived  0.0  1.0
#Sex               
#female     34  100
#male      350   63

'''
trainTktCount = train_df.groupby("Ticket")["Ticket"].transform(len)
maskGroupTrain = (trainTktCount > 1)
trainGrouped_df = train_df[maskGroupTrain]
(len(trainGrouped_df), len(trainSingle_df), len(holdoutGrouped_df), len(holdoutSingle_df))
'''
### EDA.x: Checking if child & aged people were accompanied by relatives or were they vulnerable
#Below 2 data tables clearly shows that there was NO IMPACT of Vulnerable on Suvived
'''def is_vulnerable(passenger):
    Age, SibSp, Parch = passenger
    if (((Age < 18) or (Age > 60)) and (SibSp+Parch == 0)):
        return 'vulnerable'
    else:
        return 'safe'

train_df['Vulnerable'] = train_df[['Age', 'SibSp', 'Parch']].apply(is_vulnerable, axis=1)

tab = pd.crosstab(train_df['Vulnerable'], train_df['AgeCategory'])
tab.iloc[:,:]'''
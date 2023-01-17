import xgboost as xgb

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
from pprint import pprint
# reading in dataset and viewing it
df = pd.read_csv('../input/basic_income_dataset_dalia.csv')
df.head()
# get the number of rows and columns
print(df.shape)

# get datetype info
print()
print(df.info())
# get an overview of the numeric agecolumn (.T = transposing the dataframe's order)
df.describe().T
# get an overview of all 14 object columns/features
df.describe(include='object').T
df.rename(columns = {'rural':'city_or_rural',
                     'dem_education_level':'education',
                     'dem_full_time_job':'full_time_job',
                     'dem_has_children':'has_children',
                     'question_bbi_2016wave4_basicincome_awareness':'awareness',
                     'question_bbi_2016wave4_basicincome_vote':'vote',
                     'question_bbi_2016wave4_basicincome_effect':'effect',
                     'question_bbi_2016wave4_basicincome_argumentsfor':'arg_for',
                     'question_bbi_2016wave4_basicincome_argumentsagainst':'arg_against'},
          inplace=True)
df.drop(['uuid', 'weight', 'age_group'], axis=1, inplace=True)
# new number of rows and columns
df.shape
# checking how much total missing data we have
df.isna().sum()
# in percentage: 7%
round(df['education'].isna().sum() / len(df), 3)
df.education.unique()
df.education.value_counts()
df['education'].fillna('no', inplace=True)
df.isna().sum().sum()
df.education.value_counts()
# new number of rows and columns
df.shape
# check if there are any duplicates
df.duplicated().sum()
df.drop_duplicates(keep='first', inplace=True)
# final number of rows and columns
df.shape
# recode voting
def vote_coding(row):
    if row == 'I would vote for it' : return('for')
    elif row == 'I would probably vote for it': return('for')
    elif row == 'I would vote against it': return('against')
    elif row == 'I would probably vote against it': return('against')
    elif row == 'I would not vote': return('no_action')

# apply function
df['vote'] = df['vote'].apply(vote_coding)
# drop all records who are not "for" or "against"
df = df.query("vote != 'no_action'")
df.vote.value_counts(normalize=True)
def awareness_coding(row):
    if row == 'I understand it fully': return('fully')
    elif row == 'I know something about it': return('something')
    elif row == 'I have heard just a little about it': return('little')
    elif row == 'I have never heard of it': return('nothing')

df['awareness'] = df['awareness'].apply(awareness_coding)
def effect_coding(row):
    if row == '‰Û_ stop working': return('stop_working')
    elif row == '‰Û_ work less': return('work_less')
    elif row == '‰Û_ do more volunteering work': return('volunteering_work')
    elif row == '‰Û_ spend more time with my family': return('more_family_time')
    elif row == '‰Û_ look for a different job': return('different_job')
    elif row == '‰Û_ work as a freelancer': return('freelancer')
    elif row == '‰Û_ gain additional skills': return('additional_skills')
    elif row == 'A basic income would not affect my work choices': return('no_effect')
    else: return('none_of_the_above')
    
df['effect'] = df['effect'].apply(effect_coding).astype(str)
df.age.describe(percentiles=[.2, .4, .6, .8])
def age_groups(row):
    if row <= 26: return('14_26')
    elif row <= 35: return('27_35')
    elif row <= 42: return('36_42')
    elif row <= 49: return('43_49')
    else: return('above_50')
    
df['age_group'] = df['age'].apply(age_groups)
df.drop(['age'], axis=1, inplace=True)
df['age_group'].value_counts(normalize=True).plot(kind='barh', figsize=(8,4));
arg_for = ['It reduces anxiety about financing basic needs',
           'It creates more equality of opportunity',
           'It encourages financial independence and self-responsibility',
           'It increases solidarity, because it is funded by everyone',
           'It reduces bureaucracy and administrative expenses',
           'It increases appreciation for household work and volunteering',
           'None of the above']

# count all arguments
counter = [0,0,0,0,0,0,0]

for row in df.iterrows():
    for i in range(0, len(arg_for)):
        if arg_for[i] in row[1]['arg_for'].split('|'):
            counter[i] = counter[i] + 1

# create a new dictionary 
dict_keys = ['less anxiety', 'more equality', 'financial independance', 
             'more solidarity', 'less bureaucracy', 'appreciates volunteering', 'none']

arg_dict = {}

for i in range(0, len(arg_for)):
    arg_dict[dict_keys[i]] = counter[i]

# sub-df for counted arguments
sub_df = pd.DataFrame(list(arg_dict.items()), columns=['Arguments PRO basic income', 'count'])

# plot
sub_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='Arguments PRO basic income', y='count',  
                                                      figsize=(10,6), legend=False, color='darkgrey',
                                                      title='Arguments PRO basic income')
plt.xlabel('Count'); 
df['less_anxiety'] = df['arg_for'].str.contains('anxiety')
df['more_equality'] = df['arg_for'].str.contains('equality')
arg_against = ['It is impossible to finance', 'It might encourage people to stop working',
               'Foreigners might come to my country and take advantage of the benefit',
               'It is against the principle of linking merit and reward', 
               'Only the people who need it most should get something from the state',
               'It increases dependence on the state', 'None of the above']

# count all arguments
counter = [0,0,0,0,0,0,0]

for row in df.iterrows():
    for i in range(0, len(arg_against)):
        if arg_against[i] in row[1]['arg_against'].split('|'):
            counter[i] = counter[i] + 1

# create a new dictionary 
dict_keys = ['impossible to finance', 'people stop working', 'foreigners take advantage', 
             'against meritocracy', 'only for people in need', 'more dependence on state', 'none']

arg_dict = {}

for i in range(0, len(arg_against)):
    arg_dict[dict_keys[i]] = counter[i]

# sub-df for counted arguments
sub_df = pd.DataFrame(list(arg_dict.items()), columns=['Arguments AGAINST basic income', 'count'])

# plot
sub_df.sort_values(by=['count'], ascending=True).plot(kind='barh', x='Arguments AGAINST basic income', y='count',  
                                                      figsize=(10,6), legend=False, color='darkgrey',
                                                      title='Arguments AGAINST basic income')
plt.xlabel('Count'); 
df['in_need'] = df['arg_against'].str.contains('need')
df['stop_working'] = df['arg_against'].str.contains('stop working')
df['too_costly'] = df['arg_against'].str.contains('impossible')
df.drop(['arg_for', 'arg_against'], axis=1, inplace=True)
df.head()
df.shape
df['vote'].value_counts(normalize=True).plot(kind='barh', figsize=(8,4), 
                                             color=['maroon','midnightblue']);
from statsmodels.graphics.mosaicplot import mosaic

mosaic(df, ['gender', 'vote'], gap=0.015, title='Vote vs. Gender - Mosaic Chart');
mosaic(df, ['city_or_rural', 'vote'], gap=0.015, title='Vote vs. Area - Mosaic Chart');
mosaic(df, ['full_time_job', 'vote'], gap=0.015, 
       title='Vote vs. Having a Full Time Job or not - Mosaic Chart');
mosaic(df, ['has_children', 'vote'], gap=0.015, 
       title='Vote vs. Having children or not - Mosaic Chart');
# Votes depending on having a full-time-job

sub_df = df.groupby('full_time_job')['vote'].value_counts(normalize=True).unstack()
sub_df.plot(kind='bar', color=['midnightblue', 'maroon'], figsize=(7,4))
plt.xlabel("Full Time Job")
plt.xticks(rotation=0)
plt.ylabel("Percentage of Voters\n")
plt.title('\nVote depending on having a full-time-job\n', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.2, 1.0), title='Vote');
# Votes in GERMANY and GREECE - depending on having a full-time-job

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))

# create sub-df for Germany
sub_df_1 = df[df['country_code']=='DE'].groupby('full_time_job')['vote'].value_counts(normalize=True).unstack()
sub_df_1.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax1, legend=False)
ax1.set_title('\nVotes in GERMANY depending on having a full-time-job\n', fontsize=14, fontweight='bold')
ax1.set_xlabel("Full Time Job")
ax1.set_xticklabels(labels=['No', 'Yes'], rotation=0)
ax1.set_ylabel("Percentage of Voters\n")

# create sub-df for Greece
sub_df_2 = df[df['country_code']=='GR'].groupby('full_time_job')['vote'].value_counts(normalize=True).unstack()
sub_df_2.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax2, legend=False)
ax2.set_title('\nVotes in GREECE depending on having a full-time-job\n', fontsize=14, fontweight='bold')
ax2.set_xlabel("Full Time Job")
ax2.set_xticklabels(labels=['No', 'Yes'], rotation=0)
ax2.set_ylabel("Percentage of Voters\n")

# create one legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.84, 0.85))
plt.show();
# Votes depending on education level

sub_df = df.groupby('education')['vote'].value_counts(normalize=True).unstack()
sub_df.plot(kind='bar', color = ['midnightblue','maroon'], figsize=(12,5))
plt.xlabel("Education Level")
plt.xticks(rotation=0)
plt.ylabel("Percentage of Voters\n")
plt.title('\nVote depending on education level\n', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.15, 1), title='Vote');
# Votes in GERMANY and GREECE - depending on education level

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

# create sub-df for Germany
sub_df_1 = df[df['country_code']=='DE'].groupby('education')['vote'].value_counts(normalize=True).unstack()
sub_df_1.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax1, legend=False)
ax1.set_title('\nVotes in GERMANY depending on education level\n', fontsize=14, fontweight='bold')
ax1.set_xlabel("Education Level")
ax1.set_xticklabels(labels=['High', 'Low', 'Medium', 'No'], rotation=0)
ax1.set_ylabel("Percentage of Voters\n")

# create df for Greece
sub_df_2 = df[df['country_code']=='GR'].groupby('education')['vote'].value_counts(normalize=True).unstack()
sub_df_2.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax2, legend=False)
ax2.set_title('\nVotes in GREECE depending on education level\n', fontsize=14, fontweight='bold')
ax2.set_xlabel("Education Level")
ax2.set_xticklabels(labels=['High', 'Low', 'Medium', 'No'], rotation=0)

# create one legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.83, 0.85))
plt.show();
# Votes in 4 countries - depending on education level

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14,10))

# create sub-df for Germany
sub_df_1 = df[df['country_code']=='DE'].groupby('education')['vote'].value_counts(normalize=True).unstack()
sub_df_1.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax1, legend=False)
ax1.set_title('\nVotes in GERMANY depending on education level\n', fontsize=14, fontweight='bold')
ax1.set_xlabel("Education Level")
ax1.set_xticklabels(labels=['High', 'Low', 'Medium', 'No'], rotation=0)
ax1.set_ylabel("Percentage of Voters\n")

# create sub-df for France
sub_df_2 = df[df['country_code']=='FR'].groupby('education')['vote'].value_counts(normalize=True).unstack()
sub_df_2.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax2, legend=False)
ax2.set_title('\nVotes in France depending on education level\n', fontsize=14, fontweight='bold')
ax2.set_xlabel("Education Level")
ax2.set_xticklabels(labels=['High', 'Low', 'Medium', 'No'], rotation=0)
ax2.set_ylabel("Percentage of Voters\n")

# create sub-df for Italy
sub_df_3 = df[df['country_code']=='IT'].groupby('education')['vote'].value_counts(normalize=True).unstack()
sub_df_3.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax3, legend=False)
ax3.set_title('\nVotes in Italy depending on education level\n', fontsize=14, fontweight='bold')
ax3.set_xlabel("Education Level")
ax3.set_xticklabels(labels=['High', 'Low', 'Medium', 'No'], rotation=0)

# create sub-df for Slovakia
sub_df_4 = df[df['country_code']=='SK'].groupby('education')['vote'].value_counts(normalize=True).unstack()
sub_df_4.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax4, legend=False)
ax4.set_title('\nVotes in Slovakia depending on education level\n', fontsize=14, fontweight='bold')
ax4.set_xlabel("Education Level")
ax4.set_xticklabels(labels=['High', 'Low', 'Medium', 'No'], rotation=0)

# create only one legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.95))
plt.tight_layout()
plt.show();
# Votes depending on awareness

sub_df = df.groupby('awareness')['vote'].value_counts(normalize=True).unstack()
sub_df.plot(kind='bar', color=['midnightblue', 'maroon'], figsize=(7,4))
plt.xlabel("Awareness")
plt.xticks(rotation=0)
plt.ylabel("Percentage of Voters\n")
plt.title('\nVote depending on awareness\n', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.2, 1.0), title='Vote');
# Votes depending on age

sub_df = df.groupby('age_group')['vote'].value_counts(normalize=True).unstack()
sub_df.plot(kind='bar', color=['midnightblue', 'maroon'], figsize=(9,5))
plt.xlabel("Age Group")
plt.xticks(rotation=0)
plt.ylabel("Percentage of Voters\n")
plt.title('\nVote depending on age\n', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.2, 1.0), title='Vote');
# Votes depending on effect

sub_df = df.groupby('effect')['vote'].value_counts(normalize=True).unstack()
sub_df.plot(kind='bar', color=['midnightblue', 'maroon'], figsize=(14,5))
plt.xlabel("\nEffect of Basic Income")
plt.xticks(rotation=0)
plt.ylabel("Percentage of Voters\n")
plt.title('\nVote depending on effect\n', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.1, 1.0), title='Vote');
# plot votes in 4 countries - depending on education level

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,12))

# create sub-df for those who agree/disagree with the argument:
# "It reduces anxiety about financing basic needs"
sub_df_1 = df.groupby('less_anxiety')['vote'].value_counts(normalize=True).unstack()
sub_df_1.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax1, legend=False)
ax1.set_title('\nVotes depending on attitude towards reducing_anxiety\n', fontsize=14, fontweight='bold')
ax1.set_xlabel('"It reduces anxiety about financing basic needs"')
ax1.set_xticklabels(labels=['False', 'True'], rotation=0)
ax1.set_ylabel("Percentage of Voters\n")

# create sub-df for those who agree/disagree with the argument:
# "It creates more equality of opportunity"
sub_df_2 = df.groupby('more_equality')['vote'].value_counts(normalize=True).unstack()
sub_df_2.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax2, legend=False)
ax2.set_title('\nVotes depending on attitude towards more_equality\n', fontsize=14, fontweight='bold')
ax2.set_xlabel('"It creates more equality of opportunity"')
ax2.set_xticklabels(labels=['False', 'True'], rotation=0)

# create sub-df for those who agree/disagree with the argument:
# "It might encourage people to stop working"
sub_df_3 = df.groupby('stop_working')['vote'].value_counts(normalize=True).unstack()
sub_df_3.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax3, legend=False)
ax3.set_title('\nVotes depending on attitude towards people_stop_working\n', fontsize=14, fontweight='bold')
ax3.set_xlabel('"It might encourage people to stop working"')
ax3.set_xticklabels(labels=['False', 'True'], rotation=0)
ax3.set_ylabel("Percentage of Voters\n")

# create sub-df for those who agree/disagree with the argument:
# "Only the people who need it most should get something from the state"
sub_df_4 = df.groupby('in_need')['vote'].value_counts(normalize=True).unstack()
sub_df_4.plot(kind='bar', color = ['midnightblue', 'maroon'], ax=ax4, legend=False)
ax4.set_title('\nVotes depending on attitude towards only_for_people_in_need\n', fontsize=14, fontweight='bold')
ax4.set_xlabel('"Only the people who need it most should get something from the state"')
ax4.set_xticklabels(labels=['False', 'True'], rotation=0)

# create only one legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.95))
plt.tight_layout()
plt.show();
# define our features 
features = df.drop(["vote"], axis=1)

# define our target
target = df[["vote"]]

# create dummy variables
features = pd.get_dummies(features)
print(features.shape)
features.tail(2)
print(target.shape)
# import train_test_split function
from sklearn.model_selection import train_test_split

# import LogisticRegression
from sklearn.linear_model import LogisticRegression

# import metrics
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")
# split our data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
# instantiate the logistic regression
logreg = LogisticRegression()

# train
logreg.fit(X_train, y_train)

# predict
train_preds = logreg.predict(X_train)
test_preds = logreg.predict(X_test)

# evaluate
train_accuracy_logreg = accuracy_score(y_train, train_preds)
test_accuracy_logreg = accuracy_score(y_test, test_preds)
report_logreg = classification_report(y_test, test_preds)

print("Logistic Regression")
print("------------------------")
print(f"Training Accuracy: {(train_accuracy_logreg * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_logreg * 100):.4}%")

# store accuracy in a new dataframe
score_logreg = ['Logistic Regression', train_accuracy_logreg, test_accuracy_logreg]
models = pd.DataFrame([score_logreg])
# import random forest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# create a baseline
forest = RandomForestClassifier()
# create Grid              
param_grid = {'n_estimators': [80, 100, 120],
              'criterion': ['gini', 'entropy'],
              'max_features': [5, 7, 9],         
              'max_depth': [5, 8, 10], 
              'min_samples_split': [2, 3, 4]}

# instantiate the tuned random forest
forest_grid_search = GridSearchCV(forest, param_grid, cv=3, n_jobs=-1)

# train the tuned random forest
forest_grid_search.fit(X_train, y_train)

# print best estimator parameters found during the grid search
print(forest_grid_search.best_params_)
# instantiate the tuned random forest with the best found parameters
# here I use the parameters originally got back from GridSearch in the first round
forest = RandomForestClassifier(n_estimators=120, criterion='gini', max_features=9, 
                                max_depth=10, min_samples_split=4, random_state=4)

# train the random forest
forest.fit(X_train, y_train)

# predict
train_preds = forest.predict(X_train)
test_preds = forest.predict(X_test)

# evaluate
train_accuracy_forest = accuracy_score(y_train, train_preds)
test_accuracy_forest = accuracy_score(y_test, test_preds)
report_forest = classification_report(y_test, test_preds)

print("Random Forest")
print("-------------------------")
print(f"Training Accuracy: {(train_accuracy_forest * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_forest * 100):.4}%")

# append accuracy score to our dataframe
score_forest = ['Random Forest', train_accuracy_forest, test_accuracy_forest]
models = models.append([score_forest])
# create a baseline
booster = xgb.XGBClassifier()
# create Grid
param_grid = {'n_estimators': [100],
              'learning_rate': [0.05, 0.1], 
              'max_depth': [3, 5, 10],
              'colsample_bytree': [0.7, 1],
              'gamma': [0.0, 0.1, 0.2]}

# instantiate the tuned random forest
booster_grid_search = GridSearchCV(booster, param_grid, scoring='accuracy', cv=3, n_jobs=-1)

# train the tuned random forest
booster_grid_search.fit(X_train, y_train)

# print best estimator parameters found during the grid search
print(booster_grid_search.best_params_)
# instantiate tuned xgboost
booster = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100,
                            colsample_bytree=0.7, gamma=0.1, random_state=4)

# train
booster.fit(X_train, y_train)

# predict
train_preds = booster.predict(X_train)
test_preds = booster.predict(X_test)

# evaluate
train_accuracy_booster = accuracy_score(y_train, train_preds)
test_accuracy_booster = accuracy_score(y_test, test_preds)
report_booster = classification_report(y_test, test_preds)

print("XGBoost")
print("-------------------------")
print(f"Training Accuracy: {(train_accuracy_booster * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_booster * 100):.4}%")

# append accuracy score to our dataframe
score_booster = ['XGBoost', train_accuracy_booster, test_accuracy_booster]
models = models.append([score_booster])
from sklearn import svm
# instantiate Support Vector Classification
svm = svm.SVC(kernel='rbf', random_state=4)

# train
svm.fit(X_train, y_train)

# predict
train_preds = svm.predict(X_train)
test_preds = svm.predict(X_test)

# evaluate
train_accuracy_svm = accuracy_score(y_train, train_preds)
test_accuracy_svm = accuracy_score(y_test, test_preds)
report_svm = classification_report(y_test, test_preds)

print("Support Vector Machine")
print("-------------------------")
print(f"Training Accuracy: {(train_accuracy_svm * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_svm * 100):.4}%")

# append accuracy score to our dataframe
score_svm = ['Support Vector Machine', train_accuracy_svm, test_accuracy_svm]
models = models.append([score_svm])
models
models.columns = ['Classifier', 'Training Accuracy', "Testing Accuracy"]
models.set_index(['Classifier'], inplace=True)
# sort by testing accuracy
models.sort_values(['Testing Accuracy'], ascending=[False])
print('Classification Report XGBoost: \n', report_booster)
print('------------------------------------------------------')
print('Classification Report Logistic Regression: \n', report_logreg)
print('------------------------------------------------------')
print('Classification Report SVM: \n', report_svm)
print('------------------------------------------------------')
print('Classification Report Random Forest: \n', report_forest)
from imblearn.over_sampling import SMOTE
# view previous class distribution
print(target['vote'].value_counts()) 

# resample data ONLY using training data
X_resampled, y_resampled = SMOTE().fit_sample(X_train, y_train) 

# view synthetic sample class distribution
print(pd.Series(y_resampled).value_counts()) 
# then perform ususal train-test-split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, random_state=0)
# instantiate the logistic regression
logreg2 = LogisticRegression()

# train
logreg2.fit(X_train, y_train)

# predict
train_preds = logreg2.predict(X_train)
test_preds = logreg2.predict(X_test)

# evaluate
train_accuracy_logreg2 = accuracy_score(y_train, train_preds)
test_accuracy_logreg2 = accuracy_score(y_test, test_preds)
report_logreg2 = classification_report(y_test, test_preds)

print("Logistic Regression with balanced classes")
print("------------------------")
print(f"Training Accuracy: {(train_accuracy_logreg2 * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_logreg2 * 100):.4}%")

# store accuracy in a new dataframe
score_logreg2 = ['Logistic Regression balanced', train_accuracy_logreg2, test_accuracy_logreg2]
models2 = pd.DataFrame([score_logreg2])
# instantiate the random forest with the best found parameters
forest2 = RandomForestClassifier(n_estimators=120, criterion='gini', max_features=9, 
                                 max_depth=10, min_samples_split=4, random_state=4)

# train the random forest
forest2.fit(X_train, y_train)

# predict
train_preds = forest2.predict(X_train)
test_preds = forest2.predict(X_test)

# evaluate
train_accuracy_forest2 = accuracy_score(y_train, train_preds)
test_accuracy_forest2 = accuracy_score(y_test, test_preds)
report_forest2 = classification_report(y_test, test_preds)

print("Random Forest with balanced classes")
print("-------------------------")
print(f"Training Accuracy: {(train_accuracy_forest2 * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_forest2 * 100):.4}%")

# append accuracy score to our dataframe
score_forest2 = ['Random Forest balanced', train_accuracy_forest2, test_accuracy_forest2]
models2 = models2.append([score_forest2])
# instantiate tuned xgboost
booster2 = xgb.XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100,
                            colsample_bytree=0.7, gamma=0.1, random_state=4)

# train
booster2.fit(X_train, y_train)

# predict
train_preds = booster2.predict(X_train)
test_preds = booster2.predict(X_test)

# evaluate
train_accuracy_booster2 = accuracy_score(y_train, train_preds)
test_accuracy_booster2 = accuracy_score(y_test, test_preds)
report_booster2 = classification_report(y_test, test_preds)

print("XGBoost with balanced classes")
print("-------------------------")
print(f"Training Accuracy: {(train_accuracy_booster2 * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_booster2 * 100):.4}%")

# append accuracy score to our dataframe
score_booster2 = ['XGBoost balanced', train_accuracy_booster2, test_accuracy_booster2]
models2 = models2.append([score_booster2])
from sklearn import svm
# instantiate Support Vector Classification
svm2 = svm.SVC(kernel='rbf')

# train
svm2.fit(X_train, y_train)

# predict
train_preds = svm2.predict(X_train)
test_preds = svm2.predict(X_test)

# evaluate
train_accuracy_svm2 = accuracy_score(y_train, train_preds)
test_accuracy_svm2 = accuracy_score(y_test, test_preds)
report_svm2 = classification_report(y_test, test_preds)

print("Support Vector Machine with balanced classes")
print("-------------------------")
print(f"Training Accuracy: {(train_accuracy_svm2 * 100):.4}%")
print(f"Test Accuracy:     {(test_accuracy_svm2 * 100):.4}%")

# append accuracy score to our dataframe
score_svm2 = ['SVM balanced', train_accuracy_svm2, test_accuracy_svm2]
models2 = models2.append([score_svm2])
models2
# Accuracy for balanced data
models2.columns = ['Classifier balanced', 'Training Accuracy', "Testing Accuracy"]
models2.set_index(['Classifier balanced'], inplace=True)
models2.sort_values(['Testing Accuracy'], ascending=[False])
# Accuracy for imbalanced data
models.sort_values(['Testing Accuracy'], ascending=[False])
print('Classification Report XGBoost: \n', report_booster2)
print('------------------------------------------------------')
print('Classification Report Logistic Regression: \n', report_logreg2)
print('------------------------------------------------------')
print('Classification Report SVM: \n', report_svm2)
print('------------------------------------------------------')
print('Classification Report Random Forest: \n', report_forest2)
# plot the important features - based on XGBoost
feat_importances = pd.Series(booster.feature_importances_, index=features.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='darkgrey', figsize=(10,5))
plt.xlabel('Relative Feature Importance with XGBoost');
# plot the important features - based on Random Forest
feat_importances = pd.Series(forest.feature_importances_, index=features.columns)
feat_importances.nlargest(10).sort_values().plot(kind='barh', color='darkgrey', figsize=(10,5))
plt.xlabel('Relative Feature Importance with Random Forest');

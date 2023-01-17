import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
print('Shape of dataframe:', df.shape)
df.isnull().sum()
# 1
df.columns = ['gender', 'race', 'parent_education', 'lunch', 'test_prep', 'math_score', 'reading_score', 'writing_score']

# 2
df['race'] = df.race.apply(lambda x: x[-1])

# 3
df['avg_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3

df.head()
# count of each parent_education entry
df.groupby(['parent_education']).gender.count()
df['parent_education'] = df.parent_education.apply(lambda x: 'high school' if x == 'some high school' else ('college' if x == 'some college' else x))
df.head()
fig, axs = plt.subplots(figsize=(22,6), ncols=3)
fig.subplots_adjust(wspace=0.23)

sns.scatterplot(x='math_score', y='reading_score', hue='gender', data=df, ax=axs[0])
sns.scatterplot(x='math_score', y='writing_score', hue='gender', data=df, ax=axs[1])
sns.scatterplot(x='reading_score', y='writing_score', hue='gender', data=df, ax=axs[2])
fig, axs = plt.subplots(figsize=(18,5), ncols=3)
fig.subplots_adjust(wspace=0.3)

sns.distplot(df.math_score, ax=axs[0])
sns.distplot(df.reading_score, ax=axs[1])
sns.distplot(df.writing_score, ax=axs[2])
fig, axs = plt.subplots(figsize=(15,6), ncols=3)
fig.subplots_adjust(wspace=0.5)

sns.boxplot(x='test_prep', y='math_score', data=df, ax=axs[0], fliersize=2)
sns.boxplot(x='test_prep', y='reading_score', data=df, ax=axs[1], fliersize=2)
sns.boxplot(x='test_prep', y='writing_score', data=df, ax=axs[2], fliersize=2)
fig, axs = plt.subplots(figsize=(9,7))

sns.boxplot(x='parent_education', y='avg_score', data=df, fliersize=0)
sns.swarmplot(x='parent_education', y='avg_score', data=df, color='0')
log_df = df.copy()
log_df.head()
log_df['gender'] = log_df.gender.apply(lambda x: 1 if x == 'male' else 0)
log_df['reduced_lunch'] = log_df.lunch.apply(lambda x: 1 if x == 'free/reduced' else 0)
log_df['test_prep'] = log_df.test_prep.apply(lambda x: 1 if x == 'completed' else 0)

# removing 'lunch' and 'avg_score' columns
log_df = log_df.drop(['lunch', 'avg_score'], axis=1)

log_df.head(3)
race_df = pd.get_dummies(log_df.race)
ed_df = pd.get_dummies(log_df.parent_education)
log_df = pd.concat([log_df, race_df, ed_df], axis=1)

log_df = log_df.drop(['race', 'parent_education'], axis=1)

log_df.columns = ['gender', 'test_prep', 'math_score', 'reading_score', 'writing_score', 
                  'reduced_lunch', 'race_A', 'race_B', 'race_C', 'race_D', 'race_E', 
                  'p_associates', 'p_bachelors', 'p_college', 'p_high_school', 'p_masters']

log_df.head(3)
scores = ['math_score', 'reading_score', 'writing_score']
for i in scores:
    log_df[i] = log_df[i]/100

log_df.head(3)
predictors = list(log_df.columns)
predictors.remove('gender')

X_train, X_test, y_train, y_test = train_test_split(log_df[predictors], log_df['gender'], 
                                                    test_size=0.2, random_state=38)
print('Training data:', X_train.shape, '\nTest data:', X_test.shape)
log = LogisticRegression()

# parameter space
penalty = ['l1', 'l2']
C = np.logspace(0, 4, 20)
log_params = dict(C=C, penalty=penalty)

# grid search
log_clf = GridSearchCV(log, log_params, cv=5, verbose=0)
best_log_model = log_clf.fit(X_train, y_train)
log_score = best_log_model.score(X_test, y_test)

# tuned parameters
print('Best Penalty:', best_log_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_log_model.best_estimator_.get_params()['C'])
print('\nModel Accuracy:', log_score)
log_predictions = best_log_model.predict(X_test)
pd.crosstab(y_test, log_predictions, rownames=['Actual'], colnames=['Predicted'])
logistic = LogisticRegression(penalty = best_log_model.best_estimator_.get_params()['penalty'], 
                              C = best_log_model.best_estimator_.get_params()['C'])
logistic.fit(X_train, y_train)

feature_importance = abs(np.std(X_train, 0) * list(logistic.coef_[0]))
n = len(feature_importance)

fig, axs = plt.subplots(figsize=(12,5))
sns.barplot(x = feature_importance.nlargest(n).index, 
            y = feature_importance.nlargest(n))
axs.set_xticklabels(axs.get_xticklabels(), rotation=90)
axs.set(xlabel='Feature', ylabel='Feature Importance')
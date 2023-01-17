# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# read csv.

df = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")



# add count.

df['count'] = 1



# add state_type.

df.loc[df['state'] == 'failed', 'state_type'] = 'failed'

df.loc[df['state'] == 'successful', 'state_type'] = 'successful'



# to datetime.

df['launched'] = pd.to_datetime(df['launched'])

df['deadline'] = pd.to_datetime(df['deadline'])



# add days.

df['days'] = (df['deadline'] - df['launched']).dt.days
df.head()
# show heatmap.

sns.heatmap(df.isnull())
df.groupby('main_category').sum().index
#sns.set_style('darkgrid')

mains = df.main_category.value_counts().head(15)



x = mains.values

y = mains.index



fig = plt.figure(dpi=80)

ax = fig.add_subplot(111)

ax = sns.barplot(y=y, x=x, orient='h', palette="Accent", alpha=0.8)



plt.title('Kickstarter Top 15 Category Count')

plt.show()
mains
df.groupby('category').sum().index
cats = df.category.value_counts().head(15)



x = cats.values

y = cats.index



fig = plt.figure(dpi=80)

ax = fig.add_subplot(111)

ax = sns.barplot(y=y, x=x, orient='h', palette="CMRmap", alpha=0.8)



plt.title('Kickstarter Top 15 Sub-Category Count')

plt.show()
cats
plt.style.use('seaborn-pastel')



fig, ax = plt.subplots(1, 1, dpi=100)

explode = [0,0,.1,.2, .4]

df.state.value_counts().head(5).plot.pie(autopct='%0.2f%%',

                                        explode=explode)



plt.title('Breakdown of Kickstarter Project Status')

plt.ylabel('')

plt.show()
fig, ax = plt.subplots(1, 1)

(df.backers >=1).value_counts().plot.pie(autopct='%0.0f%%', 

                                         explode=[0,.1], 

                                         labels=None, 

                                         shadow=False)



plt.ylabel('')

plt.title('Kickstarter Backer Share')

plt.legend(['backers', 'no backers'], loc=2)



plt.show()
df_failed = df[df['state'] == 'failed']

df_successful = df[df['state'] == 'successful']
df_failed.describe()
df_successful.describe()
#Plot Success Trend.



#success

df2 = df_successful[['launched', 'count']]

df2.set_index('launched', inplace=True)

df2 = df2.resample('Y').sum() 

df2



# failed

df3 = df_failed[['launched', 'count']]

df3.set_index('launched', inplace=True)

df3 = df3.resample('Y').sum() 

df3



df4 = df[['launched', 'count']]

df4 = df4[(df4['launched'] >= '2009-01-01') & (df4['launched'] < '2018-01-01')]

df4.set_index('launched', inplace=True)

df4 = df4.resample('Y').sum() 

df4



df5 = df2 / df4 * 100

df5
df5['count'].plot(label="Success rate", figsize = (8, 6))

plt.title('Success rate Trend')

plt.legend(ncol=1)

plt.show()
# Group by main_category

grouped = df[['main_category', 'state_type', 'count']].groupby(['main_category', 'state_type']).sum()



failed = grouped.xs('failed',level="state_type")["count"]

success = grouped.xs('successful',level="state_type")["count"]



index = failed.index

df1 = pd.DataFrame({'failed': failed, 'success': success}, index=index)



df1
plt.style.use('seaborn-darkgrid')

ax = df1.plot.bar(figsize = (16, 9), rot=0)

plt.title('Success and failure comparison by category')

fig = ax.get_figure()
# show Success and failure comparison by category



category = df1.index

success = df1['success'] / (df1['success'] + df1['failed'])

failed = df1['failed'] / (df1['success'] + df1['failed'])



index = failed.index

df2 = pd.DataFrame({'failed': failed, 'success': success}, index=index)



ax = df2.plot.barh(stacked = True, figsize = (8,6))



# we also need to switch the labels

plt.xlabel('Success and failure comparison by category')  

plt.ylabel('Category')

    

ind = np.arange(15)

lst = df2.values



for x, y in zip(ind, lst):

    plt.text(y[0]/2, x, '{:.2%}'.format(y[0]), ha='center', va='center')

    plt.text((y[0] + (y[1]/2)), x, '{:.2%}'.format(y[1]), ha='center', va='center')

    

plt.vlines([0.5], -1, 100, "black")

    

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

plt.show()
# add goal decile rank.

df['goal_decile_rank'] = pd.qcut(df['goal'], 10, labels=False)

df
# Group by goal_decile_rank

grouped = df[['goal_decile_rank', 'state_type', 'count']].groupby(['goal_decile_rank', 'state_type']).sum()



failed = grouped.xs('failed',level="state_type")["count"]

success = grouped.xs('successful',level="state_type")["count"]



index = failed.index

df6 = pd.DataFrame({'failed': failed, 'success': success}, index=index)



df6
plt.style.use('seaborn-darkgrid')

ax = df6.plot.bar(figsize = (16, 9), rot=0)

plt.title('Success and failure comparison by goal_decile_rank')

fig = ax.get_figure()
# show Success and failure comparison by goal_decile_rank



category = df6.index

success = df6['success'] / (df6['success'] + df6['failed'])

failed = df6['failed'] / (df6['success'] + df6['failed'])



index = failed.index

df7 = pd.DataFrame({'failed': failed, 'success': success}, index=index)



ax = df7.plot.barh(stacked = True, figsize = (8,6))



# we also need to switch the labels

plt.xlabel('Success and failure comparison by goal_decile_rank')  

plt.ylabel('decile rank')

    

ind = np.arange(15)

lst = df7.values



for x, y in zip(ind, lst):

    plt.text(y[0]/2, x, '{:.2%}'.format(y[0]), ha='center', va='center')

    plt.text((y[0] + (y[1]/2)), x, '{:.2%}'.format(y[1]), ha='center', va='center')

    

plt.vlines([0.5], -1, 100, "black")

    

ax.xaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)

plt.show()
features = df.copy()

features['success'] = np.where(features.state == 'successful', 1, 0)
df_dummy = pd.get_dummies(features['main_category'])

features = pd.concat([features.drop(['main_category'],axis=1),df_dummy],axis=1)
features['US'] = np.where(features.country=='US', 1,0)
features.drop(['ID', 'name', 'category', 'currency', 'backers', 'pledged', 'usd pledged', 'usd_pledged_real', 'usd_goal_real', 'deadline', 'launched', 'country', 'state', 'state_type', 'count'], axis=1, inplace=True)
#features = features.dropna()

features.drop(features.columns[np.isnan(features).any()], axis=1, inplace=True)
med = features['goal'].median()

MAD = 1.4826 * np.median(abs(features['goal']-med))

features = features[(med - 3 * MAD < features['goal']) & (features['goal'] < med + 3 * MAD)]
med = features['days'].median()

MAD = 1.4826 * np.median(abs(features['days']-med))

features = features[(med - 3 * MAD < features['days']) & (features['days'] < med + 3 * MAD)]
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import ensemble
X = features.drop(['success'], 1)

y = features.success



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Declare a logistic regression classifier.

# Parameter regularization coefficient C described above.

lr = LogisticRegression(penalty='l2', solver='liblinear')



# Fit the model.

fit = lr.fit(X_train, y_train)



# Display.

# print('Coefficients')

# print(fit.coef_)

# print(fit.intercept_)



pred_y_sklearn = fit.predict(X_test)



print('\n Accuracy by success')

print(pd.crosstab(pred_y_sklearn, y_test))



print('\n Percentage accuracy')

print(lr.score(X_test, y_test))



# CV

#scores = cross_val_score(lr, X, y, cv=10)



#print(scores)
def gradient_boost(estimators, depth, loss_function, sampling):

    clf = ensemble.GradientBoostingClassifier(n_estimators=estimators, 

                                              max_depth=depth, 

                                              loss=loss_function, 

                                              subsample=sampling

                                              )

    clf.fit(X_train, y_train)

    print('\n Percentage accuracy for Gradient Boosting Classifier')

    predict_train = clf.predict(X_train)

    predict_test = clf.predict(X_test)



# Accuracy tables.

    table_train = pd.crosstab(y_train, predict_train, margins=True)

    table_test = pd.crosstab(y_test, predict_test, margins=True)



    train_tI_errors = table_train.loc[0.0,1.0] / table_train.loc['All','All']

    train_tII_errors = table_train.loc[1.0,0.0] / table_train.loc['All','All']



    test_tI_errors = table_test.loc[0.0,1.0]/table_test.loc['All','All']

    test_tII_errors = table_test.loc[1.0,0.0]/table_test.loc['All','All']

    

    train_accuracy = 1 - (train_tI_errors + train_tII_errors)

    test_accuracy = 1 - (test_tI_errors + test_tII_errors)

    

    print((

    'Training set accuracy:\n'

    'Overall Accuracy: {}\n'

    'Percent Type I errors: {}\n'

    'Percent Type II errors: {}\n\n'

    'Test set accuracy:\n'

    'Overall Accuracy: {}\n'

    'Percent Type I errors: {}\n'

    'Percent Type II errors: {}'

    ).format(train_accuracy, train_tI_errors, train_tII_errors, test_accuracy, test_tI_errors, test_tII_errors))
#500 estimators, max depth of 2, loss function = 'deviance', subsampling default to 1.0

gradient_boost(500, 2, 'deviance', 1.0)
clf = ensemble.GradientBoostingClassifier(n_estimators=500, max_depth=2, loss='deviance', subsample=1.0)

clf.fit(X_train, y_train)



feature_importance = clf.feature_importances_[:30]



# Make importances relative to max importance.

plt.figure(figsize=(10,10))

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
import lightgbm as lgb

from lightgbm import LGBMClassifier



clf_lgbm = LGBMClassifier(

        n_estimators=300,

        num_leaves=15,

        colsample_bytree=.8,

        subsample=.8,

        max_depth=7,

        reg_alpha=.1,

        reg_lambda=.1,

        min_split_gain=.01

    )



clf_lgbm.fit(X_train, 

        y_train,

        eval_set= [(X_train, y_train), (X_test, y_test)], 

        eval_metric='auc', 

        verbose=0, 

        early_stopping_rounds=30

       )



acc_clf_lgbm = round(clf_lgbm.score(X_test, y_test) * 100, 2)

acc_clf_lgbm



# # Run Cross validation

# scores = cross_val_score(clf_lgbm, X, y, cv=5)

# np.mean(scores)
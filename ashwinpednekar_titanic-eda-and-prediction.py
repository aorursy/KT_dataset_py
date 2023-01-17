# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Data Wrangling & Plotting

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

import re

sns.set_style('whitegrid')

sns.set_palette("deep")



#ML

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



gender_submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train = pd.read_csv("/kaggle/input/titanic/train.csv")



df = pd.concat([train,test], axis=0, sort=False)

def display_num_nulls(df):

    '''

    Displays number of missing values in a dataframe

    

    Parameters:

    df: Dataframe

    '''

    display(pd.concat([df.isnull().sum(), df.isnull().mean()], axis=1).rename(columns={0:'sum', 1:'avg'}))



def title_extractor(df, names_col = 'Name'):

    ''' Extracts Titles from a series of names

    Parameters: 

    df: dataframe to extract from

    names_col: dolumn of df containing names

    

    Returns: series of extracted titles

    

    '''

    return df['Name'].str.extract(r'(,\s[A-z]+\.)')[0].str.strip(' ,')



def impute_category_mean(df, missing_col, by_cat_col):

    '''

    Imputes missing values using the mean defined by a particular category

    

    Parameters:

    df: dataframe

    missing_col: column to impute

    by_cat_col: column containing categories to compute mean

    

    Returns: series with imputed values

    '''

    return df.groupby(by_cat_col)[missing_col].transform(lambda x: x.fillna(x.mean()))

     
display_num_nulls(train)
display_num_nulls(test)
display_num_nulls(df)
train.isnull().mean()
sns.countplot(y=train['Survived']).set_yticklabels(['Goners', 'Survivors'])

plt.title('Count of Survivors and Goners')
sns.countplot('Sex', hue = 'Survived', data=train, palette='prism_r')

plt.title("Survivors by Gender")

plt.legend(['Dead','Survived'])
sns.countplot(hue='Survived',x='Pclass',data=train,palette = 'prism_r')

plt.title("Survivors by Passenger Class")

plt.legend(['Dead','Survived'])

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

ax = sns.distplot(train[train['Survived']==1]['Age'].dropna(),hist_kws=dict(alpha=0.7),color = 'green',bins = 30)

plt.title("Age Distribution of Survivors")

ax.set(xlabel='Age')





plt.subplot(1,2,2)

ax = sns.distplot(train[train['Survived']==0]['Age'].dropna(),hist_kws=dict(alpha=0.7),color = 'darkred',bins = 30)

plt.title("Age Distribution of Goners")

ax.set(xlabel='Age')
sns.scatterplot(y='Fare',x='Age',data=train,hue ='Survived',palette='prism_r')

plt.title("Fare and Age of passengers vs Survived")

L =plt.legend()

L.get_texts()[1].set_text('Dead')

L.get_texts()[2].set_text('Survived')

#Most passengers with lower fare died, with the exception of younger passengers
plt.hist(train['Age'].dropna(), bins=30)

plt.title("Distribution of Passenger Ages")

plt.xlabel("Age")

plt.ylabel("Count")
df['Title'] = title_extractor(df)

print("Missing values in 'Title': {}".format(df.isnull().sum()['Title']))
# train has one missing value that the title_extractor did not properly take care of

missing_title_ind = df['Title'][lambda x: pd.isnull(x)].index

display(df.loc[missing_title_ind])



# The regex in the function did not manage to capture "the Countess"

# Manually taking care of it

df.loc[df[df.Title.isnull()].index,'Title'] = 'Countess'

# Impute age by mean of each 'Title'

df['Age'] = impute_category_mean(df,'Age','Title')

# Fill in missing Fares by Passenger Class

df['Fare'] = impute_category_mean(df, 'Fare', 'Pclass')
cabin_first_letter = df['Cabin'].str[0]
pd.crosstab(df['Pclass'],cabin_first_letter.fillna('no data'))
df.drop(['Cabin'], axis=1, inplace=True)
# Since only 2 entries are missing



df['Embarked'] = df['Embarked'].fillna('mode')
display_num_nulls(df)
clf_df = pd.concat([df,pd.get_dummies(df[['Embarked','Title','Sex']], dummy_na=True, drop_first=True)], axis=1)

clf_df.drop(clf_df.select_dtypes('object').columns, axis=1, inplace=True)



# Dummied test

clf_df_t = clf_df[clf_df['Survived'].isnull()].drop('Survived', axis=1)



# Dummied train

clf_df_tr = clf_df[clf_df['Survived'].notnull()]
X = clf_df_tr.drop('Survived', axis=1)

y = clf_df_tr['Survived']

Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.3, random_state=123)



lg = LogisticRegression(solver='liblinear')

lg.fit(Xtr,ytr)

lgpred_tr = lg.predict(Xtr)

tr_score = accuracy_score(ytr, lgpred_tr)



lgpred_t = lg.predict(Xt)

t_score = accuracy_score(yt,lgpred_t)



print("Training Accuracy: {}\nTest Accuracy: {}".format("%.2f" % tr_score, "%.2f" % t_score))


rf = RandomForestClassifier(n_estimators=100, criterion='gini')

rf.fit(Xtr,ytr)

rfpred_tr = rf.predict(Xtr)

tr_score = accuracy_score(ytr, rfpred_tr)



rfpred_t = rf.predict(Xt)

t_score = accuracy_score(yt,rfpred_t)



print("Training Accuracy: {}\nTest Accuracy: {}".format("%.2f" % tr_score, "%.2f" % t_score))


svc = SVC(C=1, kernel= 'linear', gamma='auto')

svc.fit(Xtr, ytr)

svcpred_tr = svc.predict(Xtr)

svcpred_t = svc.predict(Xt)

accuracy_score(yt,svcpred_t)

print("Training Accuracy: {}\nTest Accuracy: {}".format("%.2f" % tr_score, "%.2f" % t_score))
lg = LogisticRegression(solver='liblinear')

lg.fit(X,y)

rf = RandomForestClassifier(n_estimators=100, criterion='gini')

rf.fit(X,y)

svc = SVC(kernel='linear', gamma='auto')

svc.fit(X,y)
lgpred = lg.predict(clf_df_t)

rfpred = rf.predict(clf_df_t)

svcpred = svc.predict(clf_df_t)
pd.concat([test['PassengerId'], pd.DataFrame(lgpred.astype(int))],axis=1).rename(columns={0:'Survived'}).to_csv('lg.csv', index=False)

pd.concat([test['PassengerId'], pd.DataFrame(rfpred.astype(int))],axis=1).rename(columns={0:'Survived'}).to_csv('rf.csv', index=False)

pd.concat([test['PassengerId'], pd.DataFrame(svcpred.astype(int))],axis=1).rename(columns={0:'Survived'}).to_csv('svc.csv', index=False)
pd.DataFrame([test['PassengerId'],lgpred], ['PassengerId','Survived']).transpose()
# Indices of correctly and wrongly classified samples

lg_correct_ind = Xt[yt==lgpred_t].index

lg_wrong_ind = Xt[yt!=lgpred_t].index



# Correctly classified and Wrongly classified samples

lg_correct = df.loc[lg_correct_ind]

lg_wrong = df.loc[lg_wrong_ind]

# Indices of correctly and wrongly classified samples

rf_correct_ind = Xt[yt==rfpred_t].index

rf_wrong_ind = Xt[yt!=rfpred_t].index



# Correctly classified and Wrongly classified samples

rf_correct = df.loc[rf_correct_ind]

rf_wrong = df.loc[rf_wrong_ind]

# Combination of rf and lg

wrong_ind = np.intersect1d(lg_wrong_ind, rf_wrong_ind)

wrong = df.loc[wrong_ind]
# Combination of rf and lg



correct_ind = np.intersect1d(lg_correct_ind,rf_correct_ind)

correct = df.loc[correct_ind]
fig, ax = plt.subplots(1,2, figsize=(10,3))



ax[0].hist(correct['Age'], bins=30)

ax[1].hist(wrong['Age'], bins=30)



ax[0].set_title('Age, Correctly Classified Samples')

ax[1].set_title('Age, Misclassified Samples')
fig, ax = plt.subplots(1,2, figsize=(10,3))



sns.countplot(correct['SibSp'],ax=ax[0])



sns.countplot(wrong['SibSp'],ax=ax[1])



ax[0].set_title('SibSp, Correctly Classified Samples')

ax[1].set_title('SibSp, Misclassified Samples')
fig, ax = plt.subplots(1,2, figsize=(10,3))



sns.countplot(correct['Title'].sort_values(ascending=False),ax=ax[0],)

sns.countplot(wrong['Title'].sort_values(ascending=False),ax=ax[1])





ax[0].set_title('Title, Correctly Classified Samples')

ax[1].set_title('Title, Misclassified Samples')
fig, ax = plt.subplots(1,2, figsize=(10,3))

ax[0].hist(correct['Fare'])

ax[1].hist(wrong['Fare'])
sns.set_context(font_scale=4)

sns.heatmap(confusion_matrix(yt, lgpred_t), annot=True, fmt= '.0f', cmap='coolwarm_r', annot_kws={'size':14})

plt.title('Confusion Matrix: Logistic Regression', fontsize=14)
sns.set_context(font_scale=4)

sns.heatmap(confusion_matrix(yt, rfpred_t), annot=True, fmt= '.0f', cmap='coolwarm_r', annot_kws={'size':14})

plt.title('Confusion Matrix: Random Forest', fontsize=14)
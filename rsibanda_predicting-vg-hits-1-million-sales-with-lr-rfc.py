import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import calendar

from datetime import datetime

from pandas import Series

from math import ceil
df = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv', encoding="utf-8")

dfa = df

dfa = dfa.copy()

df[:5]
cols = ['Platform', 'Developer', 'Publisher', 'Genre']



for col in cols:

    chart = df[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()

    sns.set_style("white")

    plt.figure(figsize=(12.4, 5))

    plt.xticks(rotation=90)

    sns.barplot(x=col, y='Name', data=chart[:30], palette=sns.cubehelix_palette((12 if col == 'Genre' else 30), dark=0.3, light=.85, reverse=True)).set_title(('Game count by '+col), fontsize=16)

    plt.ylabel('Count', fontsize=14)

    plt.xlabel('')
def score_group(score):

    if score >= 90:

        return '90-100'

    elif score >= 80:

        return '80-89'

    elif score >= 70:

        return '70-79'

    elif score >= 60:

        return '60-69'

    elif score >= 50:

        return '50-59'

    else:

        return '0-49'
dfh = df.dropna(subset=['Critic_Score']).reset_index(drop=True)

dfh['Score_Group'] = dfh['Critic_Score'].apply(lambda x: score_group(x))
def in_top(x):

    if x in pack:

        return x

    else:

        pass

def width(x):

    if x == 'Platform':

        return 14.4

    elif x == 'Developer':

        return 13.2

    elif x == 'Publisher':

        return 11.3

    elif x == 'Genre':

        return 13.6



def height(x):

    if x == 'Genre':

        return 8

    else:

        return 9
cols = ['Genre', 'Developer', 'Publisher', 'Platform']

for col in cols:

    pack = []

    top = dfh[['Name', col]].groupby([col]).count().sort_values('Name', ascending=False).reset_index()[:15]

    for x in top[col]:

        pack.append(x)

    dfh[col] = dfh[col].apply(lambda x: in_top(x))

    dfh_platform = dfh[[col, 'Score_Group', 'Global_Sales']].groupby([col, 'Score_Group']).median().reset_index().pivot(col, "Score_Group", "Global_Sales")

    plt.figure(figsize=(width(col), height(col)))

    sns.heatmap(dfh_platform, annot=True, fmt=".2g", linewidths=.5).set_title((' \n'+col+' vs. critic score (by median sales) \n'), fontsize=18)

    plt.ylabel('', fontsize=14)

    plt.xlabel('Score group \n', fontsize=12)

    pack = []
cols = ['Platform', 'Genre', 'Publisher', 'Developer', 'Rating']

for col in cols:

    uniques = df[col].value_counts().keys()

    uniques_dict = {}

    ct = 0

    for i in uniques:

        uniques_dict[i] = ct

        ct += 1



    for k, v in uniques_dict.items():

        df.loc[df[col] == k, col] = v
df1 = df[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]

df1 = df1.dropna().reset_index(drop=True)

df1 = df1.astype('float64')
mask = np.zeros_like(df1.corr())

mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(730, 300, sep=20, as_cmap=True, s=85, l=15, n=20) # note: 680, 350/470

with sns.axes_style("white"):

    fig, ax = plt.subplots(1,1, figsize=(15,8))

    ax = sns.heatmap(df1.corr(), mask=mask, vmax=0.2, square=True, annot=True, fmt=".3f", cmap=cmap)
fig, ax = plt.subplots(1,1, figsize=(12,5))

sns.regplot(x="Critic_Score", y="Global_Sales", data=df1, ci=None, color="#75556c", x_jitter=.02).set(ylim=(0, 17.5))
fig, ax = plt.subplots(1,1, figsize=(12,5))

sns.regplot(x="Critic_Score", y="Global_Sales", data=df1.loc[df1.Year_of_Release >= 2014],

            truncate=True, x_bins=15, color="#75556c").set(ylim=(0, 4), xlim=(50, 95))
dfb = dfa[['Name','Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]

dfb = dfb.dropna().reset_index(drop=True)

df2 = dfb[['Platform','Genre','Publisher','Year_of_Release','Critic_Score','Global_Sales']]

df2['Hit'] = df2['Global_Sales']

df2.drop('Global_Sales', axis=1, inplace=True)
def hit(sales):

    if sales >= 1:

        return 1

    else:

        return 0



df2['Hit'] = df2['Hit'].apply(lambda x: hit(x))
# Logistic regression plot with sample of the data

n = ceil(0.05 * len(df2['Hit']))

fig, ax = plt.subplots(1,1, figsize=(12,5))

sns.regplot(x="Critic_Score", y="Hit", data=df2.sample(n=n),

            logistic=True, n_boot=500, y_jitter=.04, color="#75556c")
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from sklearn import svm
df2[:5]
from pandas import get_dummies

df_copy = pd.get_dummies(df2)
df_copy[:5]
df3 = df_copy

y = df3['Hit'].values

df3 = df3.drop(['Hit'],axis=1)

X = df3.values
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.50, random_state=2)
radm = RandomForestClassifier(random_state=2).fit(Xtrain, ytrain)

y_val_1 = radm.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_1).idxmax(axis=1).values

                                   == ytest)/len(ytest))
log_reg = LogisticRegression().fit(Xtrain, ytrain)

y_val_2 = log_reg.predict_proba(Xtest)

print("Validation accuracy: ", sum(pd.DataFrame(y_val_2).idxmax(axis=1).values

                                   == ytest)/len(ytest))
all_predictions = log_reg.predict(Xtest)

print(classification_report(ytest, all_predictions))
fig, ax = plt.subplots(figsize=(3.5,2.5))

sns.heatmap(confusion_matrix(ytest, all_predictions), annot=True, linewidths=.5, ax=ax, fmt="d").set(xlabel='Predicted Value', ylabel='Expected Value')

sns.plt.title('Training Set Confusion Matrix')
indices = np.argsort(radm.feature_importances_)[::-1]



# Print the feature ranking

print('Feature ranking (top 10):')



for f in range(10):

    print('%d. feature %d %s (%f)' % (f+1 , indices[f], df3.columns[indices[f]],

                                      radm.feature_importances_[indices[f]]))
not_hit_copy = df_copy[df_copy['Hit'] == 0]
df4 = not_hit_copy

y = df4['Hit'].values

df4 = df4.drop(['Hit'],axis=1)

X = df4.values
pred = log_reg.predict_proba(X)
dfb = dfb[dfb['Global_Sales'] < 1]
dfb['Hit_Probability'] = pred[:,1]
dfb = dfb[dfb['Year_of_Release'] == 2016]

dfb.sort_values(['Hit_Probability'], ascending=[False], inplace=True)

dfb = dfb[['Name', 'Platform', 'Hit_Probability']]
dfb[:10].reset_index(drop=True)
dfb[:-11:-1].reset_index(drop=True)
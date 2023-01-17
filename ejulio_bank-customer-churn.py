# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Churn_Modelling.csv')

df.sample(10)
df.Geography = df.Geography.astype('category')

df.Gender = df.Gender.astype('category')

df.Exited = df.Exited.astype(bool)

df.HasCrCard = df.HasCrCard.astype(bool)

df.IsActiveMember = df.IsActiveMember.astype(bool)
df.dtypes
(train_df, test_df) = train_test_split(df, train_size=0.7, shuffle=True, random_state=42)
test_df.describe()
train_df.describe()
columns = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']

fig = plt.figure(figsize=(30, 5))

axes = fig.subplots(1, len(columns))



for (column, ax) in zip(columns, axes):

    exited = train_df[train_df.Exited]

    not_exited = train_df[~train_df.Exited]

    ax.hist(not_exited[column], label='Not Exited', bins=15)

    ax.hist(exited[column], label='Exited', bins=15)

    ax.set_title(f'Distribution of "{column}"')

    

fig.legend(labels=('Not Exited', 'Exited'), loc='upper center', ncol=2)
bar_width = 0.2

columns = ['Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']

fig = plt.figure(figsize=(30, 7))

axes = fig.subplots(1, len(columns))



for (column, ax) in zip(columns, axes):

    counts = train_df[~train_df.Exited][column].value_counts()

    x = counts.index.astype(int)

    ax.bar(x - bar_width / 2, counts, width=bar_width, label='Not Exited')



    counts = train_df[train_df.Exited][column].value_counts()

    x = counts.index.astype(int)

    ax.bar(x + bar_width / 2, counts, width=bar_width, label='Exited')

    

    ax.set_title(f'Counts for "{column}"')

    

fig.legend(labels=('Not Exited', 'Exited'), loc='upper center', ncol=2)
bar_width = 0.2

columns = ['Gender', 'Geography']

fig = plt.figure(figsize=(30, 7))

axes = fig.subplots(1, len(columns))



for (column, ax) in zip(columns, axes):

    counts = train_df[~train_df.Exited][column].value_counts()

    x = np.arange(len(counts.index))

    ax.bar(x - bar_width / 2, counts, width=bar_width, label='Not Exited')



    counts = train_df[train_df.Exited][column].value_counts()

    x = np.arange(len(counts.index))

    ax.bar(x + bar_width / 2, counts, width=bar_width, label='Exited')

    

    ax.set_title(f'Counts for "{column}"')

    ax.set_xticklabels(counts.index)

    ax.set_xticks(x)

    

fig.legend(labels=('Not Exited', 'Exited'), loc='upper center', ncol=2)
def cdf(values, bins):

    values = np.sort(values)

    (counts, edges) = np.histogram(values, bins=bins)

    probs = counts.cumsum() / counts.sum()

    return (edges, probs)



def plot_cdf(values, bins, title, xlabel):

    (edges, probs) = cdf(values, bins)

    plt.plot(edges[:-1], probs)

    plt.title(title)

    plt.xlabel(xlabel)
values = train_df[train_df.Exited].Age.values

plot_cdf(values, bins=20, title='CDF', xlabel='Age')

values = train_df[~train_df.Exited].Age.values

plot_cdf(values, bins=20, title='CDF', xlabel='Age')

plt.legend(labels=('Exited', 'Not Exited'))
# Perform a Smirnov-Kolmogorov Test

# https://www.statisticshowto.datasciencecentral.com/kolmogorov-smirnov-test/

# Assumption: "not exited" is the target distribution

# Null Hypothesis: "Exited age" comes from the same distribution as "not exited age" (the distributions are equal)

# Alternative hypothesis: The distributions are different



N = 100

# alpha = 0.05

critial_value = 1.36 / np.sqrt(N)



values = train_df[train_df.Exited].sample(N).Age.values

(_, cdf_exited) = cdf(values, bins=20)

values = train_df[~train_df.Exited].sample(N).Age.values

(_, cdf_not_exited) = cdf(values, bins=20)



D = np.max(cdf_not_exited - cdf_exited)



if D > critial_value:

    print(f'{D} > {critial_value} :: Reject the null hypothesis: the distributions are different')

else:

    print(f'{D} <= {critial_value} :: Accept the null hypothesis: the distributions are equal')
# Perform a Permutation Test

# Reference: Think Stats, Chapter 9 - Hypothesis Testing, p. 121

# The idea os this test is to compare two groups (g1 and g2)

# Then we simulate N random differences

# If g1 and g2 come from the same distribution, their difference should be in the 95% interval of the normal distribution

# Null Hypothesis: "Exited age" comes from the same distribution as "not exited age" (the distributions are equal)

# Alternative hypothesis: The distributions are different

exited = train_df[train_df.Exited].Age.values

not_exited = train_df[~train_df.Exited].Age.values

group_diff = exited.mean() - not_exited.mean()



N = 1000

n = len(exited)

pool = np.hstack((exited, not_exited))

diffs = []

for _ in range(N):

    np.random.shuffle(pool)

    g1 = pool[:n]

    g2 = pool[n:]

    diffs.append(g1.mean() - g2.mean())



diff_mean = np.mean(diffs)

diff_std = np.std(diffs)

if diff_mean - 2 * diff_std <= group_diff <= diff_mean + 2 * diff_std:

    print('Accept the null hypothesis: the distributions are equal')

else:

    print('Reject the null hypothesis: the distributions are different')

    

plt.hist(diffs, bins=20)

plt.axvline(group_diff)
columns = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Tenure', 'Exited']

sns.pairplot(train_df[columns], hue="Exited", plot_kws={'s': 10})
columns = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Gender']

temp_df = train_df.query('Exited == 1')[columns]

sns.pairplot(temp_df, hue='Gender', plot_kws={'s': 10})
columns = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Geography']

temp_df = train_df.query('Exited == 1')[columns]

sns.pairplot(temp_df, hue='Geography', plot_kws={'s': 10})
fig = plt.figure(figsize=(30, 5))

axes = fig.subplots(1, 3)



for (category, ax) in zip(train_df.Geography.cat.categories, axes):

    temp_df = train_df.query(f'Geography == "{category}"')

    ax.hist(temp_df.query('Exited == 0').Balance)

    ax.hist(temp_df.query('Exited == 1').Balance)

    ax.set_title(category)



plt.legend(labels=('Not Exited', 'Exited'))
def prepare_data(df):

    X = df[['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']].values

    y = df.Exited.values.astype(float)



    scaler = StandardScaler()

    scaler.fit(X)

    X = scaler.transform(X)



    X_cats = df[['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']]

    ohe = OneHotEncoder(sparse=False)

    ohe.fit(X_cats)

    X_cats = ohe.transform(X_cats)



    X = np.hstack([X, X_cats])

    return (X, y)
(X, y) = prepare_data(train_df)
X_corr = np.hstack([X, y.reshape((-1, 1))])

corr = np.corrcoef(X_corr.T)

fig = plt.figure(figsize=(20, 10))

labels = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary',

          'Geography_1', 'Geography_2', 'Geography_3', 'Gender_1', 'Gender_2',

          'HasCrCard_1', 'HasCrCard_2', 'IsActiveMember_1', 'IsActiveMember_2', 'Exited']

sns.heatmap(data=corr, annot=True, xticklabels=labels, yticklabels=labels)
(X_test, y_test) = prepare_data(test_df)
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=20, random_state=1)

rfc.fit(X, y)

y_hat = rfc.predict(X_test)



print('Precision', precision_score(y_test, y_hat))

print('Recall', recall_score(y_test, y_hat))

print('F1-score', f1_score(y_test, y_hat))

sns.heatmap(data=confusion_matrix(y_test, y_hat), annot=True, fmt='5d')
import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(rfc, random_state=1)

perm.fit(X_test, y_test)



eli5.show_weights(perm, feature_names=labels[:-1])
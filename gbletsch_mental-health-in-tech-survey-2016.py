import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
style = plt.style.available

from sklearn.preprocessing import LabelEncoder

pd.set_option('max_colwidth', 200)
pd.set_option('max_columns', 100)
url = '../input/mental-health-in-tech-2016/mental-heath-in-tech-2016_20161114.csv'
df = pd.read_csv(url)
df.shape
df.head(1)
# Duplicated data?
df.duplicated(keep=False).value_counts()
# encoding columns:

le_columns = LabelEncoder()
le_columns.fit(df.columns)
df.columns = le_columns.transform(df.columns)

# use 'le_columns.inverse_transform(['labels_here'])' to see the definition of the columns.
# set the target column

subt_columns = pd.Series(le_columns.inverse_transform(df.columns), index=df.columns)
subt_columns.loc[subt_columns == 'Have you ever sought treatment for a mental health issue from a mental health professional?']
target = 24
# standardizing the data

no_std = []
for c in df.columns:
    if len(df[c].value_counts(dropna=False)) > 10: # 10 was an arbitrary number
        no_std.append(c)
no_std
# manual inspection:

dict = zip(no_std, le_columns.inverse_transform(no_std))
for i in dict:
    print(i)
    print('Number of categories:', len(df[i[0]].value_counts(dropna=False)))
df = df.drop([55, 56, 35, 33, 34, 48, 49, 54], axis = 1)
# dropping wrong age values - (52, 'What is your age?')

a = list(df[df[52] > 90].index)
b = (list(df[df[52] < 15].index))
c = a + b
print(df[52].loc[c])
df = df.drop(c)
# column (53, 'What is your gender?')

df[53] = df[53].str.lower().str.strip()
df[53].value_counts()

m = ['Male', 'male', 'M', 'm', 'Cis Male', 'man', 'ostensibly male, unsure what that really means', 'Mail', 'Make', 
     'male (cis)', 'cis male', 'maile', 'Malr', 'Cis Man', 'Mal', 'msle', 'male.', 'sex is male', 'malr', 
     'cis man', 'mail' ]
     
f = ['Female', 'female', 'F', 'f', 'Woman', 'Femake', 'Female (cis)', 'cis female', 'woman', 'femail', 
     'cis-female/femme', 'i identify as female.', 'cis-woman', 'cisgender female', 
     'female (props for making this a freeform field, though)', 'female/woman', 'female assigned at birth' ]
df[53] = df[53].replace(m, 'm')
df[53] = df[53].replace(f, 'f')

o = list(df[53].value_counts().index)[2:]
df[53] = df[53].replace(o, 'o')
df[53].value_counts()
# Change countries to continents
# (50, 'What country do you live in?')
# (51, 'What country do you work in?')

url = '../input/countries-and-continents/continents.csv'
continents = pd.read_csv(url, usecols=['Name', 'continent'], index_col='Name', squeeze=True)

for c in continents.unique():
    l = continents[continents == c].index
    df[50] = df[50].replace(l, c)
    df[51] = df[51].replace(l, c)

# Drop rows with response = 'Other'
df = df.drop([820, 880])

df[51].value_counts(dropna=False)
# Are the floats and int already categorical?

floa = []
inte = []

for c in df.columns:
    if df[c].dtype == float:
        floa.append(c)
    elif df[c].dtype == int:
        inte.append(c)

        
print (set(zip(floa, le_columns.inverse_transform(floa))))
print (set(zip(inte, le_columns.inverse_transform(inte))))
# correcting columns types:
for c in df.columns:
    if c != 52 and c != target:
        df[c] = df[c].astype('category')
# choosing interesting data manually

age = 52
cat = [5, 7, 9, 12, 19, 32, 57, 58, 59, 60]
le_columns.inverse_transform([24])
target
# plotting graphics
plt.figure(figsize=(14, 5))
(df[target].value_counts(normalize=True) * 100).plot(kind='bar')
plt.ylabel('%')
plt.xlabel('Seek treatment')
plt.xticks(np.arange(2), ['Yes', 'No'])
plt.suptitle(str(le_columns.inverse_transform([target]))[2:-2], fontsize=16, y=1.1)
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(131)
sns.distplot(df[age], rug=True)
plt.subplot(132)
sns.distplot(df[age][df[target] == 1], rug=True, axlabel='Ages with treatment')
plt.subplot(133)
sns.distplot(df[age][df[target] == 0], rug=True, axlabel='Ages without treatment')
plt.suptitle(str(le_columns.inverse_transform([age]))[2:-2], fontsize=16, y=1.1)
plt.show()

for g, s in zip(cat, style):
    plt.figure(figsize=(14, 5))
    ax1 = plt.subplot(131)
    (df[g].value_counts(normalize=True) * 100).plot(kind='bar')
    plt.ylabel('%')
    plt.xlabel('All')
    
    ax2 = plt.subplot(132, sharey=ax1)
    (df[df[target] == 1][g].value_counts(normalize=True) * 100).plot(kind='bar')
    plt.ylabel('%')
    plt.xlabel('Seek treatment')
    
    ax3 = plt.subplot(133, sharey=ax1)
    (df[df[target] == 0][g].value_counts(normalize=True) * 100).plot(kind='bar')
    plt.ylabel('%')
    plt.xlabel("Don't seek treatment")
    
    plt.tight_layout()
    plt.suptitle(str(le_columns.inverse_transform([g]))[2:-2], fontsize=16, y=1.1)
    print()
# Make dummy variables to deal with the categorical data and the missing values
df_dummy = pd.get_dummies(df, dummy_na=False, drop_first=True)
# Making a Pearson correlation to select the most important features

corr = df_dummy.corr()[target].sort_values()[:-1]

n = 10 # n is an arbitrary number

# Plotting the Pearson correlation
plt.figure(figsize=(14, 10))
plt.suptitle('Pearson correlation', y=1.1)
plt.subplot(211)
corr.head(n).sort_values(ascending=False).plot(kind='barh')
plt.ylabel('Features')
plt.subplot(212)
corr.tail(n).sort_values(ascending=True).plot(kind='barh')
plt.xlabel('Pearson correlation')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
# Making the ML using all features
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC

# df_ml = pd.get_dummies(df, drop_first=True)
X = df_dummy.drop(target, axis=1)
y = df_dummy[target]

linear = LinearSVC(C=0.001)
score_lin = cross_val_score(linear, X, y)

rf = RandomForestClassifier()
score_rf = cross_val_score(rf, X, y)
print('rf: ', score_rf.mean(), score_rf.std())
print('linear: ', score_lin.mean(), score_lin.std())
# Random Forest using only the most important features

from heapq import nlargest

for g in range(1):
    scores = {}
    stds = {}
    for n in range(1, 100):
        X = (
            df_dummy
            .drop(target, axis=1)
            .loc[:,list(corr.head(n).index) + list(corr.tail(n).index)]
        )
        y = df_dummy[target]

        rf = RandomForestClassifier()
        score = cross_val_score(rf, X, y)
        scores[n * 2] = score.mean()
        stds[n * 2] = score.std()

    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    pd.Series(scores).plot()
    plt.xlabel('Num de features')
    plt.ylabel('Acurácia')
    
    plt.subplot(122)
    pd.Series(stds).plot()
    plt.xlabel('Num de features')
    plt.ylabel('Desvio padrão')
    plt.tight_layout()
    plt.show()
    
    largest = nlargest(10, scores, key=scores.get)
    for i in largest:
        print('O número de features é {} com {:.2f} de acurácia e desvio padrão de {:.5f}.'
              .format(i, scores[i], stds[i]))
# Linear SVC using only the most important features

for g in range(1):
    scores = {}
    stds = {}
    for n in range(1, 100):
        X = (
            df_dummy
            .drop(target, axis=1)
            .loc[:,list(corr.head(n).index) + list(corr.tail(n).index)]
        )
        y = df_dummy[target]

        clf = LinearSVC(C=0.001)
        score = cross_val_score(clf, X, y)
        scores[n * 2] = score.mean()
        stds[n * 2] = score.std()

    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    pd.Series(scores).plot()
    plt.xlabel('Num de features')
    plt.ylabel('Acurácia')
    
    plt.subplot(122)
    pd.Series(stds).plot()
    plt.xlabel('Num de features')
    plt.ylabel('Desvio padrão')
    plt.tight_layout()
    plt.show()
    
    largest = nlargest(10, scores, key=scores.get)
    for i in largest:
        print('O número de features é {} com {:.2f} de acurácia e desvio padrão de {:.5f}.'
              .format(i, scores[i], stds[i]))
    

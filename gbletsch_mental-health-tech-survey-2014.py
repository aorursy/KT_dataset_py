# Basic
import pandas as pd
import numpy as np
import os
# print(os.listdir("../input"))

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

pd.set_option('max_colwidth', 200)
df = pd.read_csv('../input/survey_2014.csv', parse_dates=True, index_col='Timestamp')
df.shape
df.info()
# correcting capitalized and spaces in columns
df.columns = df.columns.str.lower().str.strip()
df.head()
# Conferindo TS duplicados
# Just coincidence...

df.loc[df.duplicated() == True]
# dropping wrong age values

a = list(df[df.age > 100].index)
b = (list(df[df.age < 15].index))
c = a + b
df.loc[c]
df.drop(c, inplace=True)
# gender column

df.gender = df.gender.str.lower().str.strip()
df.gender.value_counts()
m = ['Male', 'male', 'M', 'm', 'Cis Male', 'Man', 'ostensibly male, unsure what that really means', 'Mail', 'Make', 
     'Male (CIS)', 'cis male', 'maile', 'Malr', 'Cis Man', 'Mal', 'msle']
     
f = ['Female', 'female', 'F', 'f', 'Woman', 'Femake', 'Female (cis)', 'Cis Female', 'woman', 'femail', 
     'cis-female/femme']
df.gender.replace(m, 'm', inplace=True)
df.gender.replace(f, 'f', inplace=True)

o = list(df.gender.value_counts().index)[2:]
df.gender.replace(o, 'o', inplace=True)
df.gender.value_counts()
# There are lots of countries, its better dealing with continens:
# I use a self made  .csv to make it. I put it here: "../input"
# I prefer the terms 'Latin America' and 'North America instead North, Central and South

continents = pd.read_csv('../input/continents.csv', usecols=['Name', 'continent'], index_col='Name', squeeze=True)
df = df.assign(continent = df.country)
df.continent = df.continent.map(continents)
df.continent.value_counts()
# Saving the cleaned df (Just in case ;-) )

df.to_csv('survey_2014_cleaned.csv')
# The meaning of the columns are confuse:

df.columns
# I made a subtitle for it:

df_sub = pd.read_csv('../input/subtitle_survey.csv', index_col='var')
df_sub
# I'm a visual guy, let's make some visualizations:
# I annotate some comments in other .csv file and putted it below the graphics.

comm = pd.read_csv('../input/comments.csv', header=None, index_col=0, squeeze=True)

for c in df.columns:
    if c in ['age']:
        plt.figure(figsize=(14, 5))
        plt.subplot(131)
        sns.distplot(df[c], rug=True)
        plt.subplot(132)
        sns.distplot(df[c][df.treatment == 'Yes'], rug=True, axlabel='Ages with treatment')
        plt.subplot(133)
        sns.distplot(df[c][df.treatment == 'No'], rug=True, axlabel='Ages without treatment')
        plt.show()
        print()
        print (comm[c])
        print('---------------')
    elif c in ['state', 'comments', 'country']: # retired from the analysis
        pass
    else:
        print()
        plt.figure(figsize=(14, 5))
        plt.subplot(132)
        sns.countplot(x=c, data=df, hue='treatment')
        plt.xticks(rotation=90)
        plt.subplot(131)
        sns.countplot(x=c, data=df)
        plt.xticks(rotation=90)
        plt.subplot(133)
        sns.countplot(x='treatment', data=df, hue=c)
        plt.xticks(rotation=90)
        plt.show()
        if c not in ['age', 'gender', 'continent']:
            print(''.join(df_sub.loc[c].values))
        
        print()
        print (comm[c])
        
# Playing with ML
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

# making dummies to deal with categoric features:
df_ml = pd.get_dummies(df.drop('comments', axis=1), drop_first=True)

X = df_ml.drop('treatment_Yes', axis=1)
y = df_ml.treatment_Yes

rf = RandomForestClassifier()
score_rf = cross_val_score(rf, X, y)
print('rf', score_rf.mean(), score_rf.std())

svc = SVC(kernel='linear')
std = StandardScaler()
X_std = std.fit_transform(X)
score_svc = cross_val_score(svc, X_std, y)
print('svc', score_svc.mean(), score_svc.std())

# Making a Pearson correlation to select the most important features

corr = df_ml.corr().treatment_Yes.sort_values()[:-1]

plt.figure(figsize=(14, 10))
plt.title('Correlação Pearson')
plt.subplot(211)
corr.head(10).sort_values(ascending=False).plot(kind='barh')
plt.xlabel('Corr Pearson')
plt.ylabel('Features')
plt.subplot(212)
corr.tail(10).sort_values(ascending=False).plot(kind='barh')
plt.xlabel('Corr Pearson')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
# Repeting RF with just the most important features:

scores_rf = {}
scores_svc = {}
for n in range(1, 50):
    X = (
        df_ml
        .drop('treatment_Yes', axis=1)
        .loc[:,list(corr.head(n).index) + list(corr.tail(n).index)]
    )
    y = df_ml.treatment_Yes

    rf = RandomForestClassifier()
    score = cross_val_score(rf, X, y)
    scores_rf[n * 2] = score.mean()

    svc = SVC(kernel='linear')
    X_std = std.fit_transform(X)
    score_svc = cross_val_score(svc, X_std, y)
    scores_svc[n * 2] = score_svc.mean()

pd.Series(scores_rf).plot()
plt.title('Random Forest')
plt.xlabel('Num de features')
plt.ylabel('Acurácia')
plt.show()
print('O melhor número de features é {} com {:.2f} de acurácia.'
      .format(max(scores_rf, key=scores_rf.get), scores_rf[max(scores_rf, key=scores_rf.get)]))

pd.Series(scores_svc).plot()
plt.title('SVC')
plt.xlabel('Num de features')
plt.ylabel('Acurácia')
plt.show()
print('O melhor número de features é {} com {:.2f} de acurácia.'
      .format(max(scores_svc, key=scores_svc.get), scores_svc[max(scores_svc, key=scores_svc.get)]))

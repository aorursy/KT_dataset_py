import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

df = pd.read_csv('../input/survey.csv')

df.info()

to_drop = ['Timestamp']  # A list of columns we will drop later on
df.Gender.unique()
# Holy hamburgers! That's a lot of possibilities for gender. Let's clean it up

# Let's see their freq counts first

df.Gender.value_counts()
df.Gender = df.Gender.str.lower()

df.Gender = df.Gender = df.Gender.replace('m', 'male')

df.Gender = df.Gender.replace('f', 'female')

df['HasMale'] = df.Gender.str.contains('male|man|guy|maile|malr|androgyne|male|mal|make|msle')

df['HasFemale'] = df.Gender.str.contains('female|woman|femail|androgyne|femake')

df['HasNB'] = df.Gender.str.contains('non-binary|enby|queer|all|fluid|agender|neuter|p')

# That's gender cleaned up.

to_drop.append('Gender')

# Moving on.

df.describe(include=['O'])
# Let's take care of country first.

df.Country.unique()
# They're clean. They can be one-hot-encoded

for country in sorted(list(df.Country.unique())):

    df['Country_'+str(country)] = (df.Country == country).astype(int)

to_drop.append('Country')
# First we need to handle the missing values in state. There are simply too many to ignore

# Let's see where exactly they are missing. I suspect that only US states have been recorded

df.groupby('Country')['state'].apply(lambda x: x.isnull().mean())
# As we can see, most countries have no state data.

# It's just easier to leave the NA's as they are

# We'll one hot them too.

for st in list(df.state.unique()):

    df['state_'+str(st)] = (df.state == st).astype(int)

to_drop.append('state')



# all the columns which are binary in nature, let's make them 01 based.

df.self_employed.fillna(df.self_employed.mode()[0], inplace=True)

for col in df.select_dtypes(include=['object']):

    u_count = len(df[col].unique()) 

    if u_count < 2:

        to_drop.append(col)

        print('adding ', col, 'to drop list as no variation')

    elif u_count == 2:

        first = list(df[col].unique())[-1]

        df[col] = (df[col] == first).astype(int)

        print('converted', col)
# Let's see what is still left

df.drop(to_drop, axis=1).info()
# For now we drop everything else.

df.work_interfere.unique()
df.work_interfere.fillna(df.work_interfere.mode().values[0], inplace=True)

df.work_interfere = df.work_interfere.map({'Never': 0, 'Rarely': 1,

                                           'Sometimes': 2, 'Often': 3})

df.no_employees.unique()
df.no_employees = df.no_employees.map({'6-25': 6, '26-100': 26,

                                       '100-500': 100, '500-1000': 500,

                                       'More than 1000': 1000, '1-5': 1

                                      })

df.benefits.unique()
# There is another pattern here. We take advantage of that:

option_map = {'Yes': 1, 'No': -1, "Don't know": 0,

              'Not sure': 0, 'Maybe': 0, 'Some of them': 0}

ynns = {'Yes': 1, 'No': -1, 'Not sure': 0}

for col in df.select_dtypes(include=['object']):

    uniques = set(df[col].unique())

    if (uniques == {'Yes', 'No', "Don't know"} or

        uniques == {'Yes', 'No', 'Not sure'} or

        uniques == {'Yes', 'No', 'Maybe'} or

        uniques == {'Yes', 'No', 'Some of them'}):

        print('encoding', col, 'To -1, 0, 1')

        df[col] = df[col].map(option_map)
df.describe(include=['O'])
df.leave.unique()
df.leave = df.leave.map({'Very easy': 0, 'Somewhat easy': 1, "Don't know": 2, 'Somewhat difficult': 3,

                         'Very difficult': 4

                        })

# this leaves comments as the only string data. Since it's quiet small in number, we'll drop it
to_drop.append('comments')

df.info()
# We obtain a clean dataset. Now we can try predicting stuff.

print(to_drop)

data = df.drop(to_drop, axis=1)

data.info()
# Thos who have shought treatment

x, y = data.drop('treatment', axis=1), data.treatment



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_jobs=-1, n_estimators=200, class_weight='balanced')

scores = cross_val_score(model, x, y, scoring='roc_auc', cv=5)

print(scores.mean())
# Family history

x, y = data.drop('family_history', axis=1), data.family_history

model = RandomForestClassifier(n_jobs=-1, n_estimators=200, class_weight='balanced')

scores = cross_val_score(model, x, y, scoring='roc_auc', cv=5)

print(scores.mean())
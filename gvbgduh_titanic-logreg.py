# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')

df_train.info()

print(df_train.head())

print('-------')

df_test = pd.read_csv('../input/test.csv')

df_test.info()

print(df_test.head())
"""

PassengerId    891 non-null int64      # Reject # Bad as feature

Survived       891 non-null int64      # Done   # Target

Pclass         891 non-null int64      # Done   # Category (3)

Name           891 non-null object     # Done   # Split and categorize title

Sex            889 non-null category   # Done   # Category (2)

Age            714 non-null float64    # Done   # Fillna with mean

SibSp          891 non-null int64      # Done   # Category ()

Parch          891 non-null int64      # Done   # Category (7)

Ticket         891 non-null object     # Done   # Two featuren: num of digits, first num. Add chars??

Fare           891 non-null float64    # Done   # Fillna with mean

Cabin          204 non-null object     # Skip   # Think of way to fillna not with 0 correlation

Embarked       889 non-null category   # Done   # Fillna with most frequent

"""

# cab = df_train.Cabin.str.extract('([A-Za-z]{1})', expand=False)

# print(cab.unique())

# print(cab.value_counts())

# print(cab.describe())

# tick = df_train.Ticket #.apply(lambda x: x.strip('0123456789./ '))

# tick = df_train.Ticket.str.extract('(\d+)', expand=False)

# tick_start_with = tick.apply(lambda x: str(x)[:2])

# tick_num_lenght = tick.apply(lambda x: len(str(x)))

# print(tick.unique())

# print(tick.value_counts())

# print(tick.describe())

# print(tick_start_with.unique())

# print(tick_start_with.value_counts())

# print(tick_start_with.describe())

# print(tick_num_lenght.unique())

# print(tick_num_lenght.value_counts())

# print(tick_num_lenght.describe())

def prepare_data(df_raw, test=False):

    df = df_raw.copy()

    # Categorize Pclass

    df.Pclass = df.Pclass.astype('category')

    print('Pclass categories: ', df.Pclass.cat.categories)

    # Name

    titles = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False).astype('category')

    cats = df['Title'].value_counts()[:9]  # [df['Title'].value_counts() > 1]

    #     print(cats)

    #     print('cats: ', [x for x, y in cats.items()])

    df['Title'] = pd.Categorical(titles, categories=[x for x, y in cats.items()])

    print('Title categories: ', df.Title.cat.categories)

    # Categorize Sex

    df.Sex = df.Sex.astype('category')

    print('Sex categories: ', df.Sex.cat.categories)

    # print('Sex categories: ', dict(enumerate(df_train.Sex.cat.categories)))

    # Fill NAN for Age using mean

    df.Age.fillna(df.Age.mean(), inplace=True)

    # Categorize SibSp

    df.SibSp = df.SibSp.astype('category')

    print('SibSp categories: ', df.SibSp.cat.categories)

    # Categorize Parch

    df.Parch = pd.Categorical(df.Parch, categories=range(10))

    print('Parch categories: ', df.Parch.cat.categories)

    # Ticket

    tick = df_train.Ticket.str.extract('(\d+)', expand=False)

    tick_start_with = tick.apply(lambda x: str(x)[:1])

    tick_num_lenght = tick.apply(lambda x: len(str(x)))

    df['TickStartWith'] = tick_start_with.astype('category')

    df['TickNumLength'] = tick_num_lenght.astype('category')

    print('TickStartWith categories: ', df.TickStartWith.cat.categories)

    print('TickNumLength categories: ', df.TickNumLength.cat.categories)

    # Fare

    df.Fare.fillna(df.Fare.mean(), inplace=True)

    # Cabin

    cab_char = df_train.Cabin.str.extract('([A-Za-z]{1})', expand=False)

    df['CabChar'] = cab_char.astype('category')

    print('CabChar categories: ', df.CabChar.cat.categories)

    # Categorize Embarked

    df.Embarked = df.Embarked.astype('category')

    df.Embarked.fillna(df.Embarked.value_counts().idxmax(), inplace=True)

    print('Embarked categories: ', df.Embarked.cat.categories)

    

    if test:

        Y_train = df.PassengerId

        df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    else:

        Y_train = df.Survived

        df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    df.info()

    return Y_train, df



Y_train, X_train = prepare_data(df_train)

Y_test, X_test = prepare_data(df_test, test=True)
"""

Playing with the Logistic Regression

"""

# from sklearn.preprocessing import LabelEncoder

# from sklearn.preprocessing import LabelBinarizer

# from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression



from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import GridSearchCV



# print(X_train.columns)

X = pd.get_dummies(X_train, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'TickStartWith', 'TickNumLength', 'CabChar'])

# 0.817090617128     # Without ticket

# 0.821591276220     # With ticket but with 2 first nums

# 0.824962204592049  # With ticket but with 1 first nums

# 0.831678527491     # With cabin char

# 0.831678527491     # With limited titles

# 0.838420242380     # With C=3

Y = Y_train

X_tr, X_t, y_tr, y_t = train_test_split(X, Y, test_size=0.3, random_state=17)



# X.info()



logreg = LogisticRegression(C=3)

cv_results = cross_val_score(logreg, X, Y, cv=5)

print('CV=5 Results: {}, Mean: {}'.format(cv_results, np.mean(cv_results)))



param_grid = {'C': [0.001, 0.01, 0.1, 0.5,  1, 3, 10, 100, 1000] }

logreg_cv = GridSearchCV(LogisticRegression(), param_grid, cv=7)

logreg_cv.fit(X, Y)

print(logreg_cv.best_params_)

print(logreg_cv.best_score_)



# TODO Scale continuous features

def scale_features(some_df):

    # Doesn't change results

    some_df.Age = (some_df.Age - some_df.Age.mean())  # / some_df.Age.var()

    some_df.Fare = (some_df.Fare - some_df.Fare.mean())  # / some_df.Fare.var()

    return some_df



# ----------------------------

# Make final Predictions

# ----------------------------

logreg = LogisticRegression(C=3)

logreg.fit(X, Y)

X_pr = pd.get_dummies(X_test, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title', 'TickStartWith', 'TickNumLength', 'CabChar'])

prediction = logreg.predict(X_pr)

print(len(Y_test), len(prediction))

s_ans = pd.concat([Y_test, pd.Series(prediction, name='Survived')], axis=1)

s_ans = s_ans.set_index('PassengerId')

# print(s_ans)

s_ans.to_csv('ans.csv')



# Res: 0.77033

# Was: 0.77511
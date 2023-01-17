# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
df
len(df)
df.loc[(df['PAY_0'] == -1) & (df['PAY_2'] == -1) & (df['PAY_3'] == -1) & (df['PAY_4'] == -1) & (df['PAY_5'] == -1) & (df['PAY_6'] == -1),'target1'] = 0

df.loc[(df['PAY_0'] >= 1)  | (df['PAY_2'] >= 1) | (df['PAY_3'] >= 1) | (df['PAY_4'] >= 1) | (df['PAY_5'] >= 1) | (df['PAY_6'] >= 1),'target1'] = 1

df.loc[((df['PAY_0'] == -1) | (df['PAY_0'] == 1)) & ((df['PAY_2'] == -1) | (df['PAY_2'] == 1)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1)),'target2'] = 0

df.loc[(df['PAY_0'] > 1)  | (df['PAY_2'] > 1) | (df['PAY_3'] > 1) | (df['PAY_4'] > 1) | (df['PAY_5'] > 1) | (df['PAY_6'] > 1),'target2'] = 1
df.loc[((df['PAY_0'] == -1) | (df['PAY_0'] == 1) | (df['PAY_0'] == 2)) & ((df['PAY_2'] == -1) | (df['PAY_2'] == 1) | (df['PAY_2'] == 2)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1) | (df['PAY_3'] == 2)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1) | (df['PAY_4'] == 2)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1) | (df['PAY_5'] == 2)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1) | (df['PAY_6'] == 2)),'target3'] = 0

df.loc[(df['PAY_0'] > 2)  | (df['PAY_2'] > 2) | (df['PAY_3'] > 2) | (df['PAY_4'] > 2) | (df['PAY_5'] > 2) | (df['PAY_6'] > 2),'target3'] = 1
print(df['target1'].value_counts())
print(df['target2'].value_counts())
print(df['target3'].value_counts())
df.loc[(df['PAY_2'] == -1) & (df['PAY_3'] == -1) & (df['PAY_4'] == -1) & (df['PAY_5'] == -1) & (df['PAY_6'] == -1),'target4'] = 0

df.loc[(df['PAY_2'] >= 1) | (df['PAY_3'] >= 1) | (df['PAY_4'] >= 1) | (df['PAY_5'] >= 1) | (df['PAY_6'] >= 1),'target4'] = 1
df.loc[((df['PAY_2'] == -1) | (df['PAY_2'] == 1)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1)),'target5'] = 0

df.loc[(df['PAY_2'] > 1) | (df['PAY_3'] > 1) | (df['PAY_4'] > 1) | (df['PAY_5'] > 1) | (df['PAY_6'] > 1),'target5'] = 1
df.loc[((df['PAY_2'] == -1) | (df['PAY_2'] == 1) | (df['PAY_2'] == 2)) & ((df['PAY_3'] == -1) | (df['PAY_3'] == 1) | (df['PAY_3'] == 2)) & ((df['PAY_4'] == -1) | (df['PAY_4'] == 1) | (df['PAY_4'] == 2)) & ((df['PAY_5'] == -1) | (df['PAY_5'] == 1) | (df['PAY_5'] == 2)) & ((df['PAY_6'] == -1) | (df['PAY_6'] == 1) | (df['PAY_6'] == 2)),'target6'] = 0

df.loc[(df['PAY_2'] > 2) | (df['PAY_3'] > 2) | (df['PAY_4'] > 2) | (df['PAY_5'] > 2) | (df['PAY_6'] > 2),'target6'] = 1
print(df['target4'].value_counts())
print(df['target5'].value_counts())
print(df['target6'].value_counts())
df.loc[(df['PAY_0'] < 1) & (df['PAY_2'] < 1) & (df['PAY_3'] < 1) & (df['PAY_4'] < 1) & (df['PAY_5'] < 1) & (df['PAY_6'] < 1),'target7'] = 0

df.loc[(df['PAY_0'] >= 1)  | (df['PAY_2'] >= 1) | (df['PAY_3'] >= 1) | (df['PAY_4'] >= 1) | (df['PAY_5'] >= 1) | (df['PAY_6'] >= 1),'target7'] = 1

df.loc[(df['PAY_0'] < 2) & (df['PAY_2'] < 2) & (df['PAY_3'] < 2) & (df['PAY_4'] < 2) & (df['PAY_5'] < 2) & (df['PAY_6'] < 2),'target8'] = 0

df.loc[(df['PAY_0'] >= 2)  | (df['PAY_2'] >= 2) | (df['PAY_3'] >= 2) | (df['PAY_4'] >= 2) | (df['PAY_5'] >= 2) | (df['PAY_6'] >= 2),'target8'] =1 
df.loc[(df['PAY_0'] < 3) & (df['PAY_2'] < 3) & (df['PAY_3'] < 3) & (df['PAY_4'] < 3) & (df['PAY_5'] < 3) & (df['PAY_6'] < 3),'target9'] = 0

df.loc[(df['PAY_0'] >= 3)  | (df['PAY_2'] >= 3) | (df['PAY_3'] >= 3) | (df['PAY_4'] >= 3) | (df['PAY_5'] >= 3) | (df['PAY_6'] >= 3),'target9'] = 1
print(df['target7'].value_counts())
print(df['target8'].value_counts())
print(df['target9'].value_counts())
df['target10'] = (df['PAY_0'] > 0).astype(int)



df['target10'].value_counts()
df
df['SEX'].value_counts()
df['SEX'] = df['SEX'] - 1
df['EDUCATION'].value_counts(sort=False)
education_dummies = pd.get_dummies(df['EDUCATION'], prefix='EDUCATION')

education_dummies = education_dummies.drop(columns=['EDUCATION_0', 'EDUCATION_1'])  # drop_first_two dos guri

df = df.join(education_dummies)

df = df.drop(columns=['EDUCATION'])



df
df['MARRIAGE'].value_counts(sort=False)
marriage_dummies = pd.get_dummies(df['MARRIAGE'], prefix='MARRIAGE')

marriage_dummies = marriage_dummies.drop(columns=['MARRIAGE_0', 'MARRIAGE_1'])  # drop_first_two dos guri

df = df.join(marriage_dummies)

df = df.drop(columns=['MARRIAGE'])



df
df['AGE'].hist()
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

import statsmodels.api as sm



target = 'target8'



maus = df[df[target] == 1.0]

bons = df[df[target] == 0.0].sample(len(maus), random_state=42)



print((maus.shape, bons.shape))



foo = maus.append(bons)

foo = foo.sample(len(foo), random_state=42)



foo['AGE'] = (foo['AGE'] >= 26).astype(int)



X = foo[foo.columns[foo.columns.str.match(r'^(SEX|EDUCATION|MARRIAGE|AGE)')]]

# X = foo[foo.columns[foo.columns.str.match(r'^(SEX|EDUCATION|AGE)')]]

# X = X.drop(columns=['EDUCATION_6'])

# X = foo[foo.columns[foo.columns.str.match(r'^(SEX|AGE)')]]

# X = foo.drop(columns=['target8'])

y = foo[target]



results = sm.OLS(y, sm.add_constant(X)).fit()



print(results.summary())



kf = KFold(n_splits=10, shuffle=True, random_state=42)

match_ratios = []



for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    model = LogisticRegression(max_iter=300)



    model.fit(X_train, y_train)

    

    predictions = model.predict(X_test)

    match_count = 0



    for index, prediction in enumerate(predictions):

        match_count += int(prediction == y_test.iloc[index])



    match_ratios.append(match_count / len(predictions))



print('Average match ratio: {:.2f}%'.format(np.mean(match_ratios) * 100))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



model = LogisticRegression(max_iter=300)



model.fit(X_train, y_train)



predictions = model.predict(X_test)

match_count = 0



for index, prediction in enumerate(predictions):

    match_count += int(prediction == y_test.iloc[index])



match_ratio = match_count / len(predictions)



print('Average match ratio: {:.2f}%'.format(match_ratio * 100))
from sklearn import tree



clf = tree.DecisionTreeClassifier(

    max_depth=3,

    min_samples_leaf=int(len(X) / 100),

#     criterion='entropy',

)



clf = clf.fit(X, y)
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt



_, ax = plt.subplots(figsize=(60, 20))



plot_tree(

    clf,

    ax=ax,

    feature_names=X.columns,

    class_names=True,

    filled=True

)



plt.show()
from sklearn.neighbors import KNeighborsClassifier



match_ratios = []



for train_index, test_index in kf.split(X):

    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    

    model = KNeighborsClassifier()



    model.fit(X_train, y_train)

    

    predictions = model.predict(X_test)

    match_count = 0



    for index, prediction in enumerate(predictions):

        match_count += int(prediction == y_test.iloc[index])



    match_ratios.append(match_count / len(predictions))



print('Average match ratio: {:.2f}%'.format(np.mean(match_ratios) * 100))
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



model = KNeighborsClassifier()



model.fit(X_train, y_train)



predictions = model.predict(X_test)

match_count = 0



for index, prediction in enumerate(predictions):

    match_count += int(prediction == y_test.iloc[index])



match_ratio = match_count / len(predictions)



print('Average match ratio: {:.2f}%'.format(match_ratio * 100))
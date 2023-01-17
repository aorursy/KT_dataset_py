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
import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib import style



style.use('fivethirtyeight')



df = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')



print(df.shape, test.shape)
df.info()
x = [col for col in df.columns]

y = []

for col in df.columns:

    i = df[col].isnull().sum()

    y.append(i)

# print(x)    

# print(y)

f = plt.subplots(figsize=(20,5))

plt.bar(x,y)
# find the corelation between features

corel = df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corel)
# plotting all the numerical features against the sales price

f, ax = plt.subplots(figsize=(20,20))

num_cols = [col for col in df.columns if df[col].dtype == 'int64' or df[col].dtype == 'float64']

pos = 0

for col in num_cols:

    pos = pos+1

    ax = plt.subplot(3, 3, pos)

    plt.plot(df['Fare'], df[col], 'o')

    plt.xlabel('Fare')

    plt.ylabel(col)    
df.head()
df['name_len'] = df['Name'].apply(lambda x : len(x))

df['Survived'].groupby(pd.qcut(df['name_len'], 5)).mean()
df['tick_len'] = df['Ticket'].apply(lambda x : len(x))

df['Survived'].groupby(pd.qcut(df['tick_len'], 4)).mean()
df['tick_let'] = df['Ticket'].apply(lambda x : x[ :1]).apply(lambda x: x.split()[0])

df['tick_let'].value_counts()
df['Name_Title'] = df['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x : x.split()[0])

df['Name_Title'].value_counts()

df['Survived'].groupby(df['Name_Title']).mean()
def name_ticket_length(train, test):

    for i in (train, test):

            i['name_len'] = i['Name'].apply(lambda x : len(x))

            i['tick_len'] = i['Ticket'].apply(lambda x: len(x))

            i['tick_lett'] = i['Ticket'].apply(lambda x : x[ :1])

            i['name_tit'] = i['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x : x.split()[0])

            

    return train, test





df = pd.read_csv('/kaggle/input/titanic/train.csv')

test1 = pd.read_csv('/kaggle/input/titanic/test.csv')

# test1 = pd.read_csv('titanic_test.csv')

test = test1.copy()



name_ticket_length(df, test)



y = df['Survived']

feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp','Parch', 

                'Fare', 'Embarked', 'name_len', 'tick_len','tick_lett', 'name_tit']

train = df[feature_cols]

test = test[feature_cols]

print(train.shape, test.shape)

from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(train, y, train_size = 0.8, test_size = 0.2, random_state = 42)

print('X - Train : ',X_train.shape)

print('Y - Train : ',y_train.shape)

print('X - Valid : ',X_valid.shape)

print('y - Valid : ',y_valid.shape)
X_train.dtypes

cat_cols = [col for col in X_train.columns if X_train[col].dtypes == 'object']

cat_cols

num_cols = [col for col in X_train.columns if (X_train[col].dtypes == 'int64') or (X_train[col].dtypes == 'float64')]

print(cat_cols)

print(num_cols)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



# processing of numerical values

num_preprocessor = SimpleImputer(strategy= 'mean')



# processing of catagorial values

cat_preprocessor = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

])

# Bundle processing

bundle_preprocess = ColumnTransformer(transformers = [

    ('nums',num_preprocessor, num_cols),

    ('cats',cat_preprocessor, cat_cols)

])





clf = Pipeline(steps = [

    ('preprocessor', bundle_preprocess),

    ('model', RandomForestClassifier(n_estimators = 400, max_features='auto', oob_score=True, random_state=1, n_jobs=-1))

])



# print(RandomForestClassifier.oob_score_)rf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

clf.fit(X_train, y_train)

prediction = clf.predict(X_valid)



cm = confusion_matrix(y_valid, prediction)

print('Confusion Matrix : \n',cm)



f1_score = f1_score(y_valid, prediction)

print('F1 Score : \n', f1_score)

score = accuracy_score(y_valid,prediction)

print('Accuracy : ', score)



# print(prediction)
test_preds = clf.predict(test)

test_preds.shape
result_titanic = pd.DataFrame({'PassengerId' : test1['PassengerId'],

                              'Survived' : test_preds })

result_titanic.head()

result_titanic.to_csv('submission.csv')
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



# processing of numerical values

num_preprocessor = SimpleImputer(strategy= 'mean')



# processing of catagorial values

cat_preprocessor = Pipeline(steps = [

    ('impute', SimpleImputer(strategy = 'most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown = 'ignore'))

])

# Bundle processing

bundle_preprocess = ColumnTransformer(transformers = [

    ('nums',num_preprocessor, num_cols),

    ('cats',cat_preprocessor, cat_cols)

])



C_range = [0.001, .050, 0.009, 0.01,0.02, 0.1]





result_table = pd.DataFrame(columns = ['C_range', 'Accuracy'])

result_table['C_range'] = C_range

j = 0

for i in C_range:

    clf = Pipeline(steps = [

        ('preprocessor', bundle_preprocess),

        ('model', LogisticRegression(penalty = 'l2',solver = 'lbfgs', C = i, max_iter = 500))

    ])

    clf.fit(X_train, y_train)

    prediction = clf.predict(X_valid)

    score = accuracy_score(y_valid,prediction)

#     print("C-Range: {} : accuracy = {}".format(i,score))

    result_table.iloc[j , 1] = score

    j +=1

result_table

# print(RandomForestClassifier.oob_score_)rf
best_clf = Pipeline(steps = [

            ('preprocessor', bundle_preprocess),

            ('model', LogisticRegression(penalty = 'l2',solver = 'lbfgs', C = 0.009, max_iter = 500))

            ])



best_clf.fit(X_train, y_train)

prediction = best_clf.predict(X_valid)

score = accuracy_score(y_valid,prediction)

print('accuracy = ',score)
logestic_prediction = best_clf.predict(test)

len(logestic_prediction)
submission = np.array(test1['PassengerId'])

submission.shape



sub_results_logesticR = pd.DataFrame({'PassengerId' : submission, 

                                      'Survived' : logestic_prediction})

# sub_results

sub_results_logesticR.to_csv('titanic_submission_logestic_reg.csv')
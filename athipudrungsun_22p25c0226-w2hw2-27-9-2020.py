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
df = pd.read_csv('/kaggle/input/titanic/train.csv')

y_test =pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
y_test.head()
y_test.describe()

# data in column age has missing value
df.isna().sum()

# missing value column ['Age', 'Cabin', 'Embarked']
df.Age.plot.hist()
df[(df.Age < 5) & (df.Survived == 1)]

# เด็กอายุต่ำกว่า 5 ปีรอด 27 คน
y_test =pd.read_csv('/kaggle/input/titanic/gender_submission.csv').set_index('PassengerId')

traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')

x_test = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')

df = pd.concat([traindf, x_test], axis=0, sort=False)

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df.info()
x = df[~(df.Survived.isna())].drop(['Survived'], axis=1)

y = df.Survived[~(df.Survived.isna())].to_frame()

y_test

x_test = df[(df.Survived.isna())].drop(['Survived'], axis=1)
y
x['y'] = 0

y['y'] = 0
for index in x.index:

    if int(index)%5 == 1:

        x.loc[index, 'y'] = 1

        y.loc[index, 'y'] = 1

    elif int(index)%5 == 2:

        x.loc[index, 'y'] = 2

        y.loc[index, 'y'] = 2

    elif int(index)%5 == 3:

        x.loc[index, 'y'] = 3

        y.loc[index, 'y'] = 3

    elif int(index)%5 == 4:

        x.loc[index, 'y'] = 4

        y.loc[index, 'y'] = 4

    elif int(index)%5 == 0:

        x.loc[index, 'y'] = 5

        y.loc[index, 'y'] = 5
x_1 = x[(x.y == 1)].drop('y', axis=1)

y_1 = y[(y.y == 1)].drop('y', axis=1)



x_2 = x[(x.y == 2)].drop('y', axis=1)

y_2 = y[(y.y == 2)].drop('y', axis=1)



x_3 = x[(x.y == 3)].drop('y', axis=1)

y_3 = y[(y.y == 3)].drop('y', axis=1)



x_4 = x[(x.y == 4)].drop('y', axis=1)

y_4 = y[(y.y == 4)].drop('y', axis=1)



x_5 = x[(x.y == 5)].drop('y', axis=1)

y_5 = y[(y.y == 5)].drop('y', axis=1)



x = x.drop('y', axis=1)
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import StandardScaler



from sklearn.pipeline import make_pipeline

from sklearn.compose import make_column_transformer

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold



# models

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier



from sklearn import metrics
onehot_testlist = ['Sex']

scalar_testlist = ['Age', 'Fare', 'SibSp', 'Parch']



col_trans=make_column_transformer(

    (OneHotEncoder(),onehot_testlist),

    (StandardScaler(), scalar_testlist),

    

#     remainder='passthrough', 

    verbose = False

)

col_trans
from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
neural_net=MLPClassifier(alpha=1, max_iter=1000)

naive_baye=GaussianNB()

tree=DecisionTreeClassifier(max_depth=5)

model_list = {'neural_net':neural_net, 'naive_baye': naive_baye, 'tree':tree}


x_test_list = [x_1, x_2, x_3, x_4, x_5]

y_test_list = [y_1, y_2, y_3, y_4, y_5]





for i in range(5):

    pipe=make_pipeline(col_trans, tree)

    ytomodel = y_test_list[i].Survived

    tree_model = pipe.fit(x_test_list[i], y_test_list[i])

    predicted = tree_model.predict(x_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted).ravel()

    precision = tp/(tp+fp)

    recall = tp/(tp+fn)

    f1 = precision*recall/(precision+recall)*2

    print(f'Dataset x_{i+1}')

    print(f'accuracy = {(tp+tn)/(tn+tp+fn+tp)}')

    print(f'precision = {precision}')

    print(f'recall = {recall}')

    print(f'f1 score = {f1}')

    print('')



Y = y.Survived

pipe=make_pipeline(col_trans, tree)

tree_model_all = pipe.fit(x, Y)

predicted_all = tree_model.predict(x_test)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted_all).ravel()

precision_all = tp/(tp+fp)

recall_all = tp/(tp+fn)

f1_all = precision_all*recall_all/(precision_all+recall_all)*2

print('')

print((tp+tn)/(tn+tp+fn+tp))

print(f'all data f1 score{f1_all}')

metrics.confusion_matrix(y_test, predicted_all)

x_test_list = [x_1, x_2, x_3, x_4, x_5]

y_test_list = [y_1, y_2, y_3, y_4, y_5]





for i in range(5):

    ytomodel = y_test_list[i].Survived

    pipe=make_pipeline(col_trans, naive_baye)

    tree_model = pipe.fit(x_test_list[i], ytomodel)

    predicted = tree_model.predict(x_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted).ravel()

    precision = tp/(tp+fp)

    recall = tp/(tp+fn)

    f1 = precision*recall/(precision+recall)*2

    print(f'Dataset x_{i+1}')

    print(f'accuracy = {(tp+tn)/(tn+tp+fn+tp)}')

    print(f'precision = {precision}')

    print(f'recall = {recall}')

    print(f'f1 score = {f1}')

    print('')

Y = y.Survived

pipe=make_pipeline(col_trans, naive_baye)

naive_baye_all = pipe.fit(x, Y)

predicted_all = naive_baye_all.predict(x_test)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted_all).ravel()

precision_all = tp/(tp+fp)

recall_all = tp/(tp+fn)

f1_all = precision_all*recall_all/(precision_all+recall_all)*2

print('')

print((tp+tn)/(tn+tp+fn+tp))

print(f'all data f1 score{f1_all}')

metrics.confusion_matrix(y_test, predicted_all)
x_test_list = [x_1, x_2, x_3, x_4, x_5]

y_test_list = [y_1, y_2, y_3, y_4, y_5]





for i in range(5):

    ytomodel = y_test_list[i].Survived

    pipe=make_pipeline(col_trans, neural_net)

    neural_model = pipe.fit(x_test_list[i], ytomodel)

    predicted = neural_model.predict(x_test)

    tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted).ravel()

    precision = tp/(tp+fp)

    recall = tp/(tp+fn)

    f1 = precision*recall/(precision+recall)*2

    print(f'Dataset x_{i+1}')

    print(f'accuracy = {(tp+tn)/(tn+tp+fn+tp)}')

    print(f'precision = {precision}')

    print(f'recall = {recall}')

    print(f'f1 score = {f1}')

    print('')



Y = y.Survived

pipe=make_pipeline(col_trans, neural_net)

neural_net_all = pipe.fit(x, Y)

predicted_all = neural_net_all.predict(x_test)

tn, fp, fn, tp = metrics.confusion_matrix(y_test, predicted_all).ravel()

precision_all = tp/(tp+fp)

recall_all = tp/(tp+fn)

f1_all = precision_all*recall_all/(precision_all+recall_all)*2

print('')

print((tp+tn)/(tn+tp+fn+tp))

print(f'all data f1 score{f1_all}')

metrics.confusion_matrix(y_test, predicted_all)
# accuracy=cross_val_score(pipe, x, y, cv=5)

# print('accuracy')

# print(f'accuracy mean {accuracy.mean()}')



trained_model = {}

y = y.Survived

for key, model in model_list.items():

    pipe=make_pipeline(col_trans, model)

    trained_model[key] = pipe.fit(x, y)

    accuracy=cross_val_score(pipe, x, y, cv=5)

    print(f'K-fold =5 accuracy of {key}', accuracy)

    print(f'accuracy of {key} mean {accuracy.mean()}')

    print('')



    
x_test
trained_model
predicted_tree = trained_model['tree'].predict(x_test)

predicted_naive = trained_model['naive_baye'].predict(x_test)

predicted_neural = trained_model['neural_net'].predict(x_test)
metrics.confusion_matrix(y_test, predicted_tree)

print('Tree', '\n', metrics.classification_report(y_test, predicted_tree), '\n')

print('Naive', '\n', metrics.classification_report(y_test, predicted_naive), '\n')

print('Neural', '\n', metrics.classification_report(y_test, predicted_neural), '\n')

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# for visualization in cell
%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import count
from IPython.display import display, Image
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
dataset = pd.concat([pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')], sort=False)
dataset.reset_index(inplace=True, drop=True)
display(dataset.head())
print('Number of instances:\n')
print('\tTrain: {}'.format(len(dataset[np.isfinite(dataset['Survived'])])))
print('\tTest: {}'.format(len(dataset[dataset['Survived'].isnull()])))
temp = dataset.copy()
temp['Sex'] = np.where(temp['Age'] <= 14.5, 'Child', np.where(temp['Sex'] == 'female', 'Woman', 'Man'))
temp['Pclass'] = temp['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})

df_survival_rate = pd.DataFrame({'Category': ['Man', 'Woman', 'Child'] * 3,
                                 'Class': ['First'] * 3 + ['Second'] * 3 + ['Third'] * 3,
                                 'Rate': [0] * 9})

for i in range(len(df_survival_rate)):
    cat, pclass = df_survival_rate.loc[i, ['Category', 'Class']]
    
    sub_df = temp[(temp['Sex'] == cat) & (temp['Pclass'] == pclass)]
    sub_df_alive = sub_df[sub_df['Survived'] == 1]
    
    df_survival_rate.loc[i, 'Rate'] = len(sub_df_alive) / len(sub_df)

f = sns.factorplot('Category', 'Rate', col='Class', data=df_survival_rate, saturation=.9, kind='bar')
_ = f.set_axis_labels('', 'Survival Rate')

del temp
dataset.drop('PassengerId', axis=1, inplace=True)
def count_nan():
    
    df = []

    for col in dataset:
        nan = dataset[dataset[col].isnull()]
        if len(nan):
            perc = round(len(nan) / len(dataset), 3) * 100
            df.append((col, len(nan), perc))
    
    df.sort(key=lambda x: x[2], reverse=True)
    df = pd.DataFrame(df, index=[el[0] for el in df], columns=[0, 'Missing', '%']).drop(0, axis=1)
    
    display(df)

count_nan()
dataset.drop('Cabin', axis=1, inplace=True)
dataset.loc[dataset['Fare'].isnull(), 'Fare'] = dataset['Fare'].mean()
dataset.loc[dataset['Embarked'].isnull(), 'Embarked'] = dataset['Embarked'].value_counts().index[0]

count_nan()
dataset = pd.concat([dataset, dataset['Name'].str.extract(r'(?P<Surname>\w+[-]?\w+),\s(?P<Title>\w+)')],
                 axis=1).drop('Name', axis=1)    

dataset.head()
info = dataset.groupby('Title').describe()['Age']['mean'].sort_values(ascending=False)
display(info)
dataset.loc[dataset['Age'].isnull(), 'Age'] = dataset.loc[dataset['Age'].isnull(), 'Title'].map(info)

count_nan()
def create_corr_mat(df, th):
    
    df2 = df.copy()

    corr_mat = df2.corr().nlargest(500, 'Survived')
    corr_mat = corr_mat[corr_mat.index]

    corr_mat = corr_mat[abs(corr_mat['Survived']) > th]
    corr_mat = corr_mat[corr_mat.index]

    return corr_mat

dataset['Sex'] = np.where(dataset['Sex'] == 'female', 1, 0)
create_corr_mat(dataset, 0)
dataset['Pclass'] = dataset['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})
child_max = dataset.groupby('Title').describe().loc['Master']['Age', 'max']
dataset['Age'] = np.where(dataset['Age'] <= child_max, 1, 0)

dataset.loc[(dataset['Sex'] == 0) & (dataset['Age'] == 1), 'Title'] = 'Master'
dataset.loc[(dataset['Sex'] == 1) & (dataset['Age'] == 1), 'Title'] = 'Miss'

dataset['Age'].groupby(dataset['Title']).mean()

#dataset.head()
dataset.tail()
#dataset['FamSize'] = dataset['SibSp'] + dataset['Parch'] + 1
#dataset.drop(['SibSp', 'Parch'], axis=1, inplace=True)

create_corr_mat(dataset, 0)

c = count(1)

dataset['FamCode'] = 0
families = dataset.groupby(['Ticket', 'Surname'])
for i, f in families:
    dataset.loc[f.index, 'FamCode'] = next(c)
    
# fixing family codes

dataset.loc[1196, 'FamCode'] = 1024
dataset[dataset['Surname'] == 'Crosby']

dataset.loc[356, 'FamCode'] = 46
dataset[dataset['Ticket'] == '113505']

families = {'0': 484, '68': 509, '104': 382, '113': 731, '136': 83, '145': 825, '175': 615, '192': 607, '267': 486,
            '352': 317, '356': 46, '371': 378, '392': 382, '417': 268, '442': 753, '451': 738, '496': 682, '529': 369,
            '532': 317, '539': 119, '556': 871, '593': 648, '627': 113, '689': 229, '704': 596, '765': 113,
            '880': 182, '892': 369, '909': 1008, '912': 733, '925': 112, '968': 91, '1012': 664, '1024': 279,
            '1041': 90, '1075': 927, '1078': 772, '1111': 959, '1129': 266, '1196': 1024, '1247': 91, '1261': 350,
            '1267': 413, '1295': 908}

for i in families:
    dataset.loc[int(i), 'FamCode'] = families[i]

dataset.head()
for i, f in dataset.groupby('FamCode'):
    dataset.loc[f.index, 'FamSize'] = len(f)
_ = sns.barplot(dataset['FamSize'], dataset['Survived'])
dataset['FamSize'] = np.where(dataset['FamSize'] == 1, 'None', np.where(dataset['FamSize'] <= 4, 'Small', 'Big'))
temp = pd.get_dummies(dataset, columns=['FamSize'])
display(create_corr_mat(temp, 0))

del temp

dataset['FamAlive'] = 0
families = dataset[dataset['FamSize'] != 'None'].groupby(['FamCode'])

for i, f in families:

    fam_al = f[f['Survived'] == 1]
    dataset.loc[f.index, 'FamAlive'] = len(fam_al) / len(f)
    
    
display(create_corr_mat(dataset, 0))
dataset.drop(['Ticket', 'Surname', 'FamCode'], axis=1, inplace=True)
final_df = pd.get_dummies(dataset)
corr_mat = create_corr_mat(final_df, 0.1)
corr_mat
final_df = final_df[corr_mat.index]
print('Number of features used: {}'.format(len(final_df.columns)))
def get_sets(df):

    xtrain = df[np.isfinite(df['Survived'])].copy()
    ytrain = xtrain['Survived'].copy()
    xtrain.drop('Survived', axis=1, inplace=True)
    xtest = df[df['Survived'].isnull()].copy()
    xtest.drop('Survived', axis=1, inplace=True)
    xtest.reset_index(inplace=True, drop=True)
    
    cols = xtrain.columns
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    xtrain = pd.DataFrame(scaler.transform(xtrain), columns=cols)
    xtest = pd.DataFrame(scaler.transform(xtest), columns=cols)
    
    return xtrain, ytrain, xtest


X_train, y_train, X_test = get_sets(final_df)
print('Train set:')
display(X_train.head())
print('Test set:')
display(X_test.head())
def run_grid_search():

    lr_values = {
                 'C': [i for i in np.arange(0.1, 1, 0.1)],
                 'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                 }

    lr = LogisticRegression(max_iter=1000)
    gs = GridSearchCV(lr, param_grid=lr_values, scoring='accuracy', return_train_score=True).fit(X_train, y_train)
    print(gs.best_params_)
    print('Mean train score: {}'.format(round(gs.cv_results_['mean_train_score'].mean(), 3)))
    print('Mean test score: {}'.format(round(gs.cv_results_['mean_test_score'].mean(), 3)))
    
    return gs.best_estimator_


best_model = run_grid_search()
preds = best_model.predict(X_test)

# To save the .csv file for submission

# csv_to_submit = np.column_stack((range(892, 1310), preds))
# np.savetxt(path_where_you_want_to_save_it, csv_to_submit, delimiter=',', header='PassengerId,Survived',
#            fmt='%i,%i', comments='')
final_df = pd.get_dummies(dataset)
corr_mat = create_corr_mat(final_df, 0.2)
display(corr_mat)
final_df = final_df[corr_mat.index]
print('Number of features used: {}'.format(len(final_df.columns)))
X_train, y_train, X_test = get_sets(final_df)

best_model = run_grid_search()
preds = best_model.predict(X_test)
dataset['Category'] = np.where(dataset['Age'] == 1, 'Child', np.where(dataset['Sex'] == 1, 'Woman', 'Man'))
final_df = pd.get_dummies(dataset)
corr_mat = create_corr_mat(final_df, 0.2)
display(corr_mat)
final_df = final_df[corr_mat.index]
print('Number of features used: {}\n'.format(len(final_df.columns)))
X_train, y_train, X_test = get_sets(final_df)

best_model = run_grid_search()
preds = best_model.predict(X_test)










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
dset = pd.concat([pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')], sort=False)
dset.reset_index(inplace=True, drop=True)
display(dset.head())
print('Number of instances:\n')
print('\tTrain: {}'.format(len(dset[np.isfinite(dset['Survived'])])))
print('\tTest: {}'.format(len(dset[dset['Survived'].isnull()])))
temp = dset.copy()
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
dset.drop('PassengerId', axis=1, inplace=True)
def count_nan():
    
    df = []

    for col in dset:
        nan = dset[dset[col].isnull()]
        if len(nan):
            perc = round(len(nan) / len(dset), 3) * 100
            df.append((col, len(nan), perc))
    
    df.sort(key=lambda x: x[2], reverse=True)
    df = pd.DataFrame(df, index=[el[0] for el in df], columns=[0, 'Missing', '%']).drop(0, axis=1)
    
    display(df)

count_nan()
dset.drop('Cabin', axis=1, inplace=True)
dset.loc[dset['Fare'].isnull(), 'Fare'] = dset['Fare'].mean()
dset.loc[dset['Embarked'].isnull(), 'Embarked'] = dset['Embarked'].value_counts().index[0]

count_nan()
dset = pd.concat([dset, dset['Name'].str.extract(r'(?P<Surname>\w+[-]?\w+),\s(?P<Title>\w+)')],
                 axis=1).drop('Name', axis=1)    

dset.head()
info = dset.groupby('Title').describe()['Age']['mean'].sort_values(ascending=False)
display(info)
dset.loc[dset['Age'].isnull(), 'Age'] = dset.loc[dset['Age'].isnull(), 'Title'].map(info)

count_nan()
def create_corr_mat(df, th):
    
    df2 = df.copy()

    corr_mat = df2.corr().nlargest(500, 'Survived')
    corr_mat = corr_mat[corr_mat.index]

    corr_mat = corr_mat[abs(corr_mat['Survived']) > th]
    corr_mat = corr_mat[corr_mat.index]

    return corr_mat

dset['Sex'] = np.where(dset['Sex'] == 'female', 1, 0)
create_corr_mat(dset, 0)
dset['Pclass'] = dset['Pclass'].map({1: 'First', 2: 'Second', 3: 'Third'})
child_max = dset.groupby('Title').describe().loc['Master']['Age', 'max']
dset['Age'] = np.where(dset['Age'] <= child_max, 1, 0)

dset.head()
dset['Age'].groupby(dset['Title']).mean()
dset.loc[(dset['Sex'] == 0) & (dset['Age'] == 1), 'Title'] = 'Master'
dset.loc[(dset['Sex'] == 1) & (dset['Age'] == 1), 'Title'] = 'Miss'

dset['Age'].groupby(dset['Title']).mean()
dset['FamSize'] = dset['SibSp'] + dset['Parch'] + 1
dset.drop(['SibSp', 'Parch'], axis=1, inplace=True)

create_corr_mat(dset, 0)
c = count(1)

dset['FamCode'] = 0
families = dset.groupby(['Ticket', 'Surname'])
for i, f in families:
    dset.loc[f.index, 'FamCode'] = next(c)

dset.head()
counter = 0
for i, f in dset[dset['FamSize'] > 1].groupby('FamCode'):
    if len(f) == 1 and counter < 3:
        display(f)
        counter += 1
    elif len(f) == 1 and counter >= 3:
        counter += 1
        
print('There are {} instances that need to be corrected.'.format(counter))
dset[dset['Surname'] == 'Crosby']
dset.loc[1196, 'FamCode'] = 1024
dset[dset['Surname'] == 'Crosby']
dset.loc[356, 'FamCode'] = 46
dset[dset['Ticket'] == '113505']
families = {'0': 484, '68': 509, '104': 382, '113': 731, '136': 83, '145': 825, '175': 615, '192': 607, '267': 486,
            '352': 317, '356': 46, '371': 378, '392': 382, '417': 268, '442': 753, '451': 738, '496': 682, '529': 369,
            '532': 317, '539': 119, '556': 871, '593': 648, '627': 113, '689': 229, '704': 596, '765': 113,
            '880': 182, '892': 369, '909': 1008, '912': 733, '925': 112, '968': 91, '1012': 664, '1024': 279,
            '1041': 90, '1075': 927, '1078': 772, '1111': 959, '1129': 266, '1196': 1024, '1247': 91, '1261': 350,
            '1267': 413, '1295': 908}

for i in families:
    dset.loc[int(i), 'FamCode'] = families[i]
for i, f in dset.groupby('FamCode'):
    dset.loc[f.index, 'FamSize'] = len(f)
_ = sns.barplot(dset['FamSize'], dset['Survived'])
dset['FamSize'] = np.where(dset['FamSize'] == 1, 'None', np.where(dset['FamSize'] <= 4, 'Small', 'Big'))
temp = pd.get_dummies(dset, columns=['FamSize'])
display(create_corr_mat(temp, 0))

del temp
dset['FamAlive'] = 0
families = dset[dset['FamSize'] != 'None'].groupby(['FamCode'])

for i, f in families:

    fam_al = f[f['Survived'] == 1]
    dset.loc[f.index, 'FamAlive'] = len(fam_al) / len(f)
    
    
display(create_corr_mat(dset, 0))
dset.head()
dset.drop(['Ticket', 'Surname', 'FamCode'], axis=1, inplace=True)
final_df = pd.get_dummies(dset)
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
final_df = pd.get_dummies(dset)
corr_mat = create_corr_mat(final_df, 0.2)
display(corr_mat)
final_df = final_df[corr_mat.index]
print('Number of features used: {}'.format(len(final_df.columns)))
X_train, y_train, X_test = get_sets(final_df)

best_model = run_grid_search()
preds = best_model.predict(X_test)
dset['Category'] = np.where(dset['Age'] == 1, 'Child', np.where(dset['Sex'] == 1, 'Woman', 'Man'))
final_df = pd.get_dummies(dset)
corr_mat = create_corr_mat(final_df, 0.2)
display(corr_mat)
final_df = final_df[corr_mat.index]
print('Number of features used: {}\n'.format(len(final_df.columns)))
X_train, y_train, X_test = get_sets(final_df)

best_model = run_grid_search()
preds = best_model.predict(X_test)
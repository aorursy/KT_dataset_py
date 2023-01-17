import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy import stats
filepath = '../input/titanic/train.csv'
titanic_train = pd.read_csv(filepath)
titanic_train.head()
titanic_train.info()
data = []
for i in titanic_train.columns:
    if i == 'PassengerId' or i == 'Ticket' or i == 'Name' or i == 'Cabin':
        role = 'id'
    elif i == 'Survived':
        role = 'outcome'
    else:
        role = 'variable'
        
    if i == 'Survived':
        level = 'binary'
    elif i == 'Pclass'or i == 'SibSp' or i == 'Parch':
        level = 'ordinal'
    elif i == 'Fare' or i == 'Age':
        level = 'interval'
    else:
        level = 'nominal'
    
    keep = True
    if role == 'id':
        keep = False
        
    dtype = titanic_train[i].dtype
        
    i_dict = {
        'varname': i,
        'role': role,
        'level': level,
        'keep': keep,
        'dtype': dtype
    }
    data.append(i_dict);

meta = pd.DataFrame(data, columns = ['varname', 'role', 'level', 'keep', 'dtype'])
meta.set_index('varname', inplace = True)
meta
v = meta[(meta.keep == True) & (meta.level == 'interval')].index
titanic_train[v].describe()
v = meta[(meta.keep == True) & (meta.level == 'ordinal')].index
titanic_train[v].describe()
v = meta[(meta.keep == True) & (meta.level == 'binary')].index
titanic_train[v].describe()
for i in titanic_train.columns:
    missing_values = titanic_train[i].isnull().sum()
    perc_missing_values = missing_values / titanic_train.shape[0]
    if missing_values > 0:
        print('There are {} missing values ({:.2%}) in {}'
              .format(missing_values, perc_missing_values, i));
drop_index = titanic_train[titanic_train.Embarked.isnull()].index
titanic_train.drop(drop_index, inplace = True)
titanic_train.shape[0]
mean_imputer = SimpleImputer(strategy = 'mean')
titanic_train.Age = mean_imputer.fit_transform(titanic_train[['Age']]).ravel()
titanic_train.Age.isnull().any()
v = meta[(meta.keep == True) & (meta.level == 'nominal')].index
for i in v:
    plt.figure()
    f, ax = plt.subplots(figsize = (10, 8))
    data = titanic_train[[i, 'Survived']]
    sns.barplot(data = data, 
                x = i,
                y = 'Survived',
                ax = ax)
    plt.show();
v = meta[(meta.keep == True) & (meta.level == 'ordinal')].index
for i in v:
    plt.figure()
    f, ax = plt.subplots(figsize = (10, 8))
    data = titanic_train[[i, 'Survived']]
    sns.barplot(data = data, 
                x = i,
                y = 'Survived',
                ax = ax)
    plt.show();
v = meta[(meta.keep == True) & (meta.level == 'interval')].index
for i in v:
    plt.figure()
    f, ax = plt.subplots(figsize = (10, 8))
    ax.set_title(i)
    data = titanic_train[i].plot.hist()
    plt.show();
def corr_heatmap(data, factors):
    plt.figure()
    f, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(data = data[factors].corr(),
                vmax = 0.5,
                annot = True,
                fmt = '.2f',
                square = True)
    plt.show();
v = meta[meta.keep == True].index
corr_heatmap(data = titanic_train,
             factors = v)
titanic_train.to_csv('cleaned_titanic_train.csv', index = False)
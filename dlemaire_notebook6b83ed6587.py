import re

# to handle datasets
import pandas as pd
import numpy as np

# for visualization
import matplotlib.pyplot as plt

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LogisticRegression

# to evaluate the models
from sklearn.metrics import accuracy_score, roc_auc_score

# to persist the model and the scaler
import joblib

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)
# load the data - it is available open source and online

data = pd.read_csv('https://www.openml.org/data/get_csv/16826755/phpMYEkMl')

# display data
data.head()
# replace interrogation marks by NaN values

data = data.replace('?', np.nan)
# retain only the first cabin if more than
# 1 are available per passenger

def get_first_cabin(row):
    try:
        return row.split()[0]
    except:
        return np.nan
    
data['cabin'] = data['cabin'].apply(get_first_cabin)
# extracts the title (Mr, Ms, etc) from the name variable

def get_title(passenger):
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
data['title'] = data['name'].apply(get_title)
# cast numerical variables as floats

data['fare'] = data['fare'].astype('float')
data['age'] = data['age'].astype('float')
# drop unnecessary variables

data.drop(labels=['name','ticket', 'boat', 'body','home.dest'], axis=1, inplace=True)

# display data
data.head()
# save the data set

data.to_csv('titanic.csv', index=False)
target = 'survived'
vars_num = [var for var in data.columns if data[var].dtypes != 'O']
vars_cat = [var for var in data.columns if data[var].dtypes == 'O']

print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))
# first in numerical variables

vars_with_na = [
    var for var in vars_num
    if data[var].isnull().sum() > 0 
]

# print percentage of missing values per variable
data[vars_with_na].isnull().mean()
data
# now in categorical variables
vars_with_naO = [
    var for var in vars_cat
    if data[var].isnull().sum() > 0 
]

# print percentage of missing values per variable
data[vars_with_naO].isnull().mean()

data[vars_with_naO].nunique()
def analyse_continuous(df, var):
    df = df.copy()
    df[var].hist(bins=30)
    plt.ylabel('Number of vars')
    plt.xlabel(var)
    plt.title(var)
    plt.show()


for var in vars_num:
    analyse_continuous(data, var)
X_train, X_test, y_train, y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
x=X_train
def get_first_cabin_letter(row):
    try:
        return row[0]
    except:
        return np.nan
    
X_train['cabin'] = x['cabin'].apply(get_first_cabin_letter)
X_train['cabin'].unique()
for var in vars_with_na:

    # calculate the mode using the train set
    mode_val = X_train[var].mode()[0]
    print ("%s : %s " %( var, mode_val ) )
    # add binary missing indicator (in train and test)
    X_train[var+'_na'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var+'_na'] = np.where(X_test[var].isnull(), 1, 0)

    # replace missing values by the mode
    # (in train and test)
    X_train[var] = X_train[var].fillna(mode_val)
    X_test[var] = X_test[var].fillna(mode_val)

# check that we have no more missing values in the engineered variables
X_train[vars_with_na].isnull().sum()
for var in vars_with_naO:

    # calculate the mode using the train set
    
    # add binary missing indicator (in train and test)
    X_train[var+'_na'] = np.where(X_train[var].isnull(), 1, 0)
    X_test[var+'_na'] = np.where(X_test[var].isnull(), 1, 0)

    # replace missing values by the mode
    # (in train and test)
    X_train[var] = X_train[var].fillna("Missing")
    X_test[var] = X_test[var].fillna("Missing")

# check that we have no more missing values in the engineered variables
X_train[vars_with_naO].isnull().sum()
#X_train
#vars_with_naO
def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the lines in the dataset

    df = df.copy()

    tmp = df.groupby(var)['age'].count() / len(df)

    return tmp[tmp > rare_perc].index


for var in vars_cat:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.05)
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(
        frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(
        frequent_ls), X_test[var], 'Rare')
X_train
def replace_categories(train, test, var):

    # order the categories in a variable from that with the lowest
    # house sale price, to that with the highest
    ordered_labels = train.groupby(var)[target].mean().sort_values().index

    # create a dictionary of ordered categories to integer values
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

    # use the dictionary to replace the categorical strings by integers
    train[var] = train[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)
x=X_train
for var in vars_cat:
    ordered_labels = x.groupby(var)["age"].mean().sort_values().index
    print (ordered_labels)
    for w in ordered_labels:
        x[var+w] = np.where(x[var]==w, 1, 0)
    
X_train=x
X_train.head()
x=X_test
for var in vars_cat:
    ordered_labels = x.groupby(var)["age"].mean().sort_values().index
    print (ordered_labels)
    for w in ordered_labels:
        x[var+w] = np.where(x[var]==w, 1, 0)
    
X_test=x
X_test.head()
X_train.drop(columns=['sex','cabin','embarked','title'], inplace=True)
X_test.drop(columns=['sex','cabin','embarked','title'], inplace=True)
X_train['fare'].unique()
# capture all variables in a list

vars_trn = ['pclass', 'age', 'sibsp', 'parch', 'fare']
# count number of variables
print (vars_trn)

# create scaler
scaler = StandardScaler()

#  fit  the scaler to the train set
scaler.fit(X_train[vars_trn]) 

# transform the train and test set
X_train[vars_trn] = scaler.transform(X_train[vars_trn])

X_test[vars_trn] = scaler.transform(X_test[vars_trn])
# Remember to set the seed (random_state for this sklearn function)
clf = LogisticRegression(random_state=0, C=0.0005)
clf.fit(X_train, y_train)

X_train.head()
X_test.head()
X_test['cabinC',"embarkedC"]=0
# set up the model

y_pred=clf.predict(X_test)
auc=roc_auc_score(y_test, y_pred)
acc=accuracy_score(y_test, y_pred)
sco=clf.score(X_test, y_test)
print (auc,acc,sco)
from joblib import dump, load
dump(clf, 'LRModel.joblib')
# let's now save the train and test sets for the next notebook!

X_train.to_csv('xtrain.csv', index=False)
X_test.to_csv('xtest.csv', index=False)

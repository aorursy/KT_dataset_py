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

data = pd.read_csv('../input/phpMYEkMl.csv')

# display data
data.head()
#getting to know all the columns
data.columns
#shape of the data
data.shape
#analysing cabin columns
data['cabin'].values[:5]
type(data['cabin'].values[0])
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
data = pd.read_csv("titanic.csv")
data.head()
data.info()
vars_num =['survived','age','sibsp','parch','fare'] # fill your code here
vars_cat =['pclass','sex','cabin','embarked','title'] # fill your code here
# for column in data.columns:
#     if(data[column].dtype == object):
#         vars_cat.append(column)
#     else:
#         vars_num.append(column)
target = 'survived'
data.columns
print('Number of numerical variables: {}'.format(len(vars_num)))
print('Number of categorical variables: {}'.format(len(vars_cat)))
vars_num
vars_cat
# first in numerical variables
data_num = data[vars_num]
data_num.head()
len(data)
data_num.isna().mean()
# now in categorical variables
data_cat = data[vars_cat]
data_cat.head()
data_cat.isna().mean()
def analyse_cat(df, var):
    df = df.copy()
    df[var].value_counts().plot.bar()
    plt.title(var)
    plt.xlabel('attributes')
    plt.ylabel('No of passengers')
    plt.show()
for col in vars_cat:
    print(25*'-'+col+'-'*25)
    print(data[col].value_counts())
    print(len(data[col].value_counts()))
    analyse_cat(data, col)
def analyse_continuous(df, var):
    df = df.copy()
    df[var].hist(bins=30)
    plt.ylabel('Number of passengers')
    plt.xlabel(var)
    plt.title(var)
    plt.show()
for col in vars_num:
    analyse_continuous(data, col)
X_train, X_test, Y_train, Y_test = train_test_split(
    data.drop('survived', axis=1),  # predictors
    data['survived'],  # target
    test_size=0.2,  # percentage of obs in test set
    random_state=0)  # seed to ensure reproducibility

X_train.shape, X_test.shape
X_train.head()
Y_train.value_counts()
Y_train[:5]
X_train[vars_cat].head()
vars_num =['age','sibsp','parch','fare']
X_train[vars_num].head()
#for numerical values
X_train[vars_num].isna().sum()
for var in vars_num:
    X_train[var].fillna(data[var].median(),inplace=True)
    X_test[var].fillna(data[var].median(),inplace=True)
X_train[vars_num].isna().sum()
#for categorical values
X_train[vars_cat].isna().sum()
X_train[vars_cat] = X_train[vars_cat].fillna('missing')
X_test[vars_cat] = X_test[vars_cat].fillna('missing')
X_train[vars_cat].isna().sum()
X_train.isna().sum()
X_test.isna().sum()
X_train.cabin.value_counts()
X_train['cabin'] = X_train['cabin'].apply(lambda x: 'missing' if x == 'missing' else x[0])
X_test['cabin'] = X_test['cabin'].apply(lambda x: 'missing' if x == 'missing' else x[0])
X_train.cabin.value_counts()
X_test.cabin.value_counts()

vars_cat
for col in vars_cat:
    print(25*'-'+col+'-'*25)
    print(X_train[col].value_counts())
    print(25*'.')
    print(X_test[col].value_counts())
    print(len(X_train[col].value_counts()))
temp = X_train.sex.value_counts() 
list(temp[temp > 200].index)
def find_frequent_labels(df, var, rare_perc):
    
    # function finds the labels that are shared by more than
    # a certain % of the houses in the dataset

    df = df.copy()
    temp = df[var].value_counts() / len(df)
    return list(temp[temp > rare_perc].index)



for var in vars_cat:
    
    # find the frequent categories
    frequent_ls = find_frequent_labels(X_train, var, 0.05)
    
    # replace rare categories by the string "Rare"
    X_train[var] = np.where(X_train[var].isin(frequent_ls), X_train[var], 'Rare')
    
    X_test[var] = np.where(X_test[var].isin(frequent_ls), X_test[var], 'Rare')
for col in vars_cat:
    print(25*'-'+col+'-'*25)
    print(X_train[col].value_counts())
    print(25*'.')
    print(X_test[col].value_counts())
    print(len(X_train[col].value_counts()))

from sklearn.preprocessing import OneHotEncoder
X_train = pd.get_dummies(X_train, columns=vars_cat, drop_first=True)
X_test = pd.get_dummies(X_test, columns=vars_cat, drop_first=True)
X_train.columns
X_test.columns
X_train.head()
X_train.drop(columns=['embarked_Rare'],inplace=True)
len(X_train.columns)
len(X_test.columns)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train.head()
X_train = pd.DataFrame(scaler.fit_transform(X_train),columns = X_train.columns)
X_train.head()
X_test = pd.DataFrame(scaler.fit_transform(X_test),columns = X_test.columns)
X_test.head()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
logreg = LogisticRegression(C=0.05,n_jobs=-1 ,random_state=0)
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(accuracy_score(Y_test,Y_pred))
print(roc_auc_score(Y_test,logreg.predict_proba(X_test)[:,1]))

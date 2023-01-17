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
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier

%matplotlib inline
pd.options.display.max_columns = None
test_raw = pd.read_csv("/kaggle/input/titanic/test.csv")
train_raw = pd.read_csv("/kaggle/input/titanic/train.csv")
train_raw.head()
# Let's check for missing values, we will have to address these missing values later 
null_cols = set(train_raw.columns[train_raw.isna().any()].tolist())
null_cols.update(test_raw.columns[test_raw.isna().any()].tolist())
print(null_cols)
# Let's first check the easier label columns 
print(train_raw['Survived'].groupby(train_raw['Pclass']).mean(), '\n')
print(train_raw['Survived'].groupby(train_raw['Sex']).mean(), '\n')
print(train_raw['Survived'].groupby(train_raw['Embarked']).mean())

# We can see that Pclass, Sex, and Embarking Port were very important
# for determining survival 
# Now let's look at numeric columns, but first we must cut them
def get_age_range(data):
    bins = [0, 18, 35, 50, np.inf]
    names = ['0-18', '18-35', '35-50', '50+']
    return pd.cut(data['Age'], bins, labels=names)

train_raw['age_range'] = get_age_range(train_raw)
print(train_raw[['Survived','age_range']].groupby(['age_range']).mean())
# print(train_raw[['Survived','age_range','Pclass']].groupby(['age_range','Pclass']).mean())

del train_raw['age_range'] # let's clean that up 

# Looks like being young is an advantage to survive the titanic 
# What about your family size?

def get_family_size(data):
    return data.apply(lambda row: 
                      "alone" if (row["SibSp"] + row["Parch"]) == 0 else "small" 
                      if (row["SibSp"] + row["Parch"]) <= 3 
                      else "large", axis=1 )

train_raw["family_size"] = get_family_size(train_raw)
        

print(train_raw[['Survived','family_size']].groupby(['family_size']).mean())
del train_raw['family_size']

# Small families had higher survival rates, followed by solo travelers, and finally large families 
# What about Fare paid?

train_raw['fare_range'] = pd.qcut(train_raw['Fare'], 6)
print(train_raw[['Survived','fare_range']].groupby(['fare_range']).mean())
del train_raw['fare_range']

# Of course the more you paid, the richer you were, the more likely you survived
# For cabin, let's separate the number and letter

def get_cabin_letter(data):
    return data.apply(lambda row: "Missing" if str(row['Cabin'])[0] == "n" else str(row['Cabin'])[0], axis=1)

def get_cabin_num(data):
    raw_num = data['Cabin'].apply(lambda x: str(x).split(' ')[-1][1:])
    raw_num.replace('an', np.NaN, inplace = True)
    return raw_num.apply(lambda x: int(x) if not pd.isnull(x) and x != '' else np.NaN)

train_raw['cabin_letter'] = get_cabin_letter(train_raw)
print(train_raw[['Survived','cabin_letter']].groupby(['cabin_letter']).mean().sort_values(by='Survived',ascending=False)
      , '\n')
del train_raw['cabin_letter']

# We see that D, E, B cabins had higher survival rates, and those with missing cabins had the lowest
# But what about cabin numbers?

train_raw['cabin_num'] = pd.qcut(get_cabin_num(train_raw), 5)
print(train_raw[['Survived','cabin_num']].groupby(['cabin_num']).mean().sort_values(by='Survived',ascending=False))
del train_raw['cabin_num']

# Looks like certain cabin numbers also had higher survival rates
# Let's do something similar to ticket
def get_ticket_len(data):
    return data.apply(lambda row: len(row['Ticket']), axis=1)

def get_ticket_letter(data):
    return data.apply(lambda row: str(str(row['Ticket'])[0]), axis=1)

def map_ticket_letter(letter):
    if letter in ['3','2','1','S','P','C','A']:
        return letter
    elif letter in ['W','4','7','F','6','L','5','8','9']:
        return "uncommon" 
    else:
        return "unknown"

# Does the ticket length tell us anything?
train_raw['ticket_len'] = get_ticket_len(train_raw)
print(train_raw[['Survived','ticket_len']].groupby(['ticket_len']).mean(), '\n')
del train_raw['ticket_len']
#Looking at the results, a bit

#What about the first ticket letter?
train_raw['ticket_letter'] = get_ticket_letter(train_raw)
print(train_raw[['Survived','ticket_letter']].groupby(['ticket_letter']).mean().sort_values(by='Survived',ascending=False), '\n')
#There's definitely some info here, but we may have to cut down on which ticket letter matter, by looking at the counts
print(train_raw[['Survived','ticket_letter']].groupby(['ticket_letter']).count().sort_values(by='Survived',ascending=False), '\n')

del train_raw['ticket_letter']
# Let's examine the info we can extract from Name

def get_title(data):
    return data.apply(lambda row: row['Name'].split(",")[1].strip().split(" ")[0], axis=1)

def get_name_len(data):
    return data.apply(lambda row: len(row['Name']), axis=1)

# We can get the title from Name
train_raw['title'] = get_title(train_raw)
print(train_raw[['Survived','title']].groupby(['title']).mean().sort_values(by='Survived',ascending=False))
del train_raw['title']
# We can see certain nobility titles seem to have way better odds of survival

# What about how long the name is?
train_raw['name_len'] = pd.qcut(get_name_len(train_raw), 5)
print(train_raw[['Survived','name_len']].groupby(['name_len']).mean().sort_values(by='Survived',ascending=False))
del train_raw['name_len']
# We see that long names have higher survival rates, maybe people with longer names were also richer? 
# These functions transform our columns into the final form we need them in 
def trans_name(train, test):
    for data in [train, test]:
            data["title"] = get_title(data)
            data["name_len"] = get_name_len(data)
            del data["Name"]
    return train,test

def trans_age(train, test):
    for data in [train, test]:
        data["age_missing"] = data.apply(lambda row: 1 if row['Age'] != row['Age'] else 0, axis=1)
        newAges = train.groupby(['title', 'Pclass'])['Age']
        data['Age'] = newAges.transform(lambda x: x.fillna(x.mean()))
    return train,test
    
def trans_ticket(train, test):
    for data in [train, test]:
        data["ticket_len"] = get_ticket_len(data)
        data["ticket_letter"] = get_ticket_letter(data)
        data["ticket_letter"] = data.apply(lambda row: map_ticket_letter(row['ticket_letter']), axis=1)
        del data['Ticket']
    return train, test

def trans_family(train, test):
    for data in [train, test]:
        data["family_size"] = get_family_size(data)
        del data["SibSp"]
        del data["Parch"]
    return train,test

def trans_fare(train, test):
    mean = train['Fare'].mean()    
    train['Fare'].fillna(mean, inplace=True) 
    test['Fare'].fillna(mean, inplace=True) 
    return train,test

def trans_embarked(train, test):
    train['Embarked'] = train['Embarked'].fillna("S") 
    test['Embarked'] = test['Embarked'].fillna("S") 
    return train,test

def trans_cabin(train, test):
    for data in [train, test]:
        data["cabin_missing"] = data.apply(lambda row: 0 if row['Cabin'] == row['Cabin'] else 1, axis=1)
        data["cabin_letter"] = get_cabin_letter(data)
        data["cabin_num1"] = get_cabin_num(data)
        data['cabin_num'] = pd.qcut(train['cabin_num1'],5)
    
    train = pd.concat((train, pd.get_dummies(train['cabin_num'], prefix = 'cabin_num')), axis = 1)
    test = pd.concat((test, pd.get_dummies(test['cabin_num'], prefix = 'cabin_num')), axis = 1)
    del train['cabin_num']
    del test['cabin_num']
    del train['cabin_num1']
    del test['cabin_num1']
    del test['Cabin']
    del train['Cabin']
    return train, test

def get_dummies(train, test):
    cols_to_expand = ['Sex','Embarked', 'Pclass', 'title', 'cabin_letter','family_size', 'ticket_letter']
    for column in cols_to_expand:
        vals = set(train[column].unique())
        vals = vals.intersection(set(test[column].unique()))
        new_cols = [column + "_" + str(val) for val in vals]

        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[new_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[new_cols]), axis = 1)
    
        del train[column]
        del test[column]

    return train, test
def get_cols_by_type(data):
    colsbytype = {}
    for idx, val in zip(data.dtypes.index, data.dtypes.values):
        if idx == 'Survived':
            continue
        val = str(val)
        curr = colsbytype.get(val, set())
        curr.add(idx)
        colsbytype[val] = curr
    for key in colsbytype.keys():
        columns = list(colsbytype[key])
        columns.sort()
        colsbytype[key] = columns
    return colsbytype

def scale_data(train, test, cols):
    cols = list(train.columns)
    cols.remove("PassengerId")
    cols.remove("Survived")
    all_cols = get_cols_by_type(test[cols])
    num_cols = all_cols['int64']
    num_cols.extend(all_cols['float64'])

    scaler = MinMaxScaler()
    X_train = train.copy()
    X_test = test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test

train = train_raw.copy()
test = test_raw.copy()

train, test = trans_name(train, test)
train, test = trans_age(train, test)  
train, test = trans_ticket(train, test)
train, test = trans_family(train, test)
train, test = trans_fare(train, test)
train, test = trans_embarked(train, test)
train, test = trans_cabin(train, test)
train, test = get_dummies(train, test)
train, test = scale_data(train, test, list(train.columns))

print("Columns: ", len(train.columns))
cols = list(train.columns[2:]) # features relevant to our ML model

print("Any Nulls: ", train.isnull().values.any(), test.isnull().values.any())
train.head()
# We will use these functions to select the best features from our data
def select_cols(feature_cols, target, data, k):
    from sklearn.feature_selection import SelectKBest, f_classif

    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(data[feature_cols], data[target]) 

    selected_features = pd.DataFrame(selector.inverse_transform(X_new), index=data.index, columns=feature_cols)
    selected_columns = selected_features.columns[selected_features.var() != 0]
    return selected_columns
    
def find_best_cols(cols, target, data):
    import warnings
    warnings.filterwarnings("ignore")
    state = 1993  
    size = 0.30 
   
    X_train = data[:500]
    X_valid = data[500:]
    
    lr = ensemble.GradientBoostingRegressor() #Base Model
    lr.fit(X_train[cols], X_train[target].values.ravel())
    print ("Base Score: ", lr.score(X_train[cols], X_train[target].values.ravel())) 
    best_score = 0
    best_cols = cols
    for k in range(len(cols)//4, len(cols)):
        lr = ensemble.RandomForestClassifier() # NOTE: using different classifier yields diff results 
        curr_cols = select_cols(cols, target, X_train, k)
        lr.fit(X_train[curr_cols], X_train[target].values.ravel())
        os_score = lr.score(X_valid[curr_cols], X_valid[target].values.ravel())
        if os_score > best_score:
            is_score = lr.score(X_train[curr_cols], X_train[target].values.ravel())
            print ("K= ", k, ", IS score: ", is_score, ", OS score: ", os_score)
            best_score = os_score
            best_cols = curr_cols
            
    return best_cols

best_cols = find_best_cols(cols, "Survived", train)
best_cols
print("Previous Columns: ", len(best_cols))
print("Selected Columns: ", len(cols))
cols = best_cols
# We'll use this function to save our results to CSV
def save_results(model, data):
    pred_test = model.predict(data)

    #PassengerId,Survived
    test_res = test[["PassengerId"]].copy()
    test_res["Survived"] = pred_test
    test_res.to_csv("/kaggle/working/my_predictions.csv", index=False)
    return test_res
# We use this function to tune our model
def get_tuned_model(estimator, param_grid, scoring, X_train, Y_train):
    from sklearn.model_selection import GridSearchCV

    grid = GridSearchCV(estimator = estimator, 
                       param_grid = param_grid,
                       scoring = scoring,
                       cv=3,
                       n_jobs= -1
                      )

    tuned = grid.fit(X_train, Y_train)

    print ("Best score: ", tuned.best_score_) 
    print ("Best params: ", tuned.best_params_)
    print ("IS Score: ", tuned.score(X_train, Y_train)) 
    
    return tuned
param_grid = { 
    "learning_rate":  [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],
    "n_estimators": [32, 64, 100, 200, 400, 500],
}

gbc = ensemble.GradientBoostingClassifier()
gbc_tuned = get_tuned_model(gbc, param_grid, "accuracy", train[cols], train[['Survived']].values.ravel())
# save_results(gbc_tuned, test[cols]) # Uncomment whichever model you want to use
param_grid = { 
    "criterion" : ["gini", "entropy"], 
    "min_samples_leaf" : [1, 5, 10], 
#     "min_samples_split" : [2, 4, 10, 12, 16],
#     "n_estimators": n_estimators,
    'max_leaf_nodes': range(4,20)
}

forest = RandomForestClassifier()
ft_tuned = get_tuned_model(forest, param_grid, "accuracy", train[cols], train[['Survived']].values.ravel())
# Best score:  0.8293955181721173
# Best params:  {'criterion': 'gini', 'max_leaf_nodes': 14, 'min_samples_leaf': 1}
# IS Score:  0.856341189674523
save_results(ft_tuned, test[cols])
logit = LogisticRegression()

param_grid = {'penalty': ['l1','l2'], 
              'C': [0.001,0.01,0.1,1,10,100],
              'max_iter': [100,200,300,500]
             }

log_tuned = get_tuned_model(logit, param_grid, "accuracy", train[cols], train[['Survived']].values.ravel())
# save_results(log_tuned, test[cols])

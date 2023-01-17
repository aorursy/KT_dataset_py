

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

import matplotlib.pylab as plt



from sklearn import ensemble

from sklearn.preprocessing import MinMaxScaler

from sklearn.inspection import permutation_importance



%matplotlib inline

pd.options.display.max_columns = None
test_raw = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train_raw = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train_raw.head()
train_raw = train_raw[train_raw['SalePrice'] < 450000] # filtering what i think are outliers 



train_raw['MSSubClass'] = train_raw.apply(lambda row: str(row['MSSubClass']), axis=1) # I'm gonna treat these as dummies not as numerical values

test_raw['MSSubClass'] = test_raw.apply(lambda row: str(row['MSSubClass']), axis=1)
train_raw['SalePrice'].hist(bins=100, figsize=(15, 5))
bins = [110000, 200000]

groups = train_raw.groupby(np.digitize(train_raw['SalePrice'], bins))



groups.mean() 
groups.nunique()
def get_quality_cols(data):

    quality_cols = data.columns[data.isin(['Ex','Po']).any()]  #'Gd', 'TA', 'Fa', 

    return quality_cols



def get_cols_by_type(data):

    colsbytype = {}

    for idx, val in zip(data.dtypes.index, data.dtypes.values):

        if idx == 'SalePrice':

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



def map_quality_values(train,test):

    qual_map = {'Ex': 5, 'Gd': 4, 'TA':3, 'Fa': 2, 'Po':1, 'NA':0}

    for data in [train,test]:

        quality_cols = get_quality_cols(data)  

        data[quality_cols] = data[quality_cols].replace(qual_map)

    return train, test



def clean_data(train_raw, test_raw):

    train = train_raw.copy()

    test = test_raw.copy()

    for data in [train, test]:

        colsbytype = get_cols_by_type(data)

        data[colsbytype['object']] = data[colsbytype['object']].fillna('NA')

        data[colsbytype['int64']] = data[colsbytype['int64']].fillna(0)

        data[colsbytype['float64']] = data[colsbytype['float64']].fillna(0.0)

    return train,test

def get_age_info(train,test):

    for data in [train,test]:

        data['age'] = data['YrSold'] - data['YearBuilt'] 

        data['facelift'] = data['YearRemodAdd'] - data['YearBuilt']



        del data['YrSold']

        del data['YearBuilt']

        del data['YearRemodAdd']



    return train, test    



def get_baths(train, test):

    for data in [train,test]:

        data['bathrooms'] = data['FullBath'] + (0.5 * data['HalfBath']) + data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath'])

        

        del data['FullBath']

        del data['HalfBath']

        del data['BsmtFullBath'] 

        del data['BsmtHalfBath']

    

    return train, test



def get_size(train,test):

    for data in [train,test]:

        data['has_2ndFl'] = data.apply(lambda row: 1 if row['2ndFlrSF'] > 1 else 0, axis=1)

        del data["1stFlrSF"]

        del data["2ndFlrSF"]

    return train, test 

    

def expand_dummies(train, test):

    cols = get_cols_by_type(train)

    for data in [train,test]:

        for col in cols['object']:

            vals = set(train[col].unique())

            vals.update(test[col].unique())

            for val in vals:

                data[col + "_" + val] = data.apply(lambda row: 1 if row[col] == val else 0, axis=1)

    for col in cols['object']:

        for data in [train,test]:

            del data[col]

    return train, test
def scale_data(train, test, cols):

    cols = list(train.columns)

    cols.remove("Id")

    cols.remove("SalePrice")

    all_cols = get_cols_by_type(test[cols])

    num_cols = all_cols['int64']

    num_cols.extend(all_cols['float64'])



    scaler = MinMaxScaler()

    X_train = train.copy()

    X_test = test.copy()

    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])

    X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_test

train, test = clean_data(train_raw, test_raw)

train, test = map_quality_values(train,test)

train, test = get_age_info(train, test)

train, test = get_baths(train, test)

train, test = get_size(train, test)

train, test = expand_dummies(train, test)

train, test = scale_data(train, test, list(train.columns))



print("Any Nulls: ", train.isnull().values.any(), test.isnull().values.any())

print(len(train.columns), len(test.columns) )

train.head()
cols = list(train.columns) # cols will hold our features 

cols.remove("Id")

cols.remove("SalePrice")
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

   

    X_train = data[:1000]

    X_valid = data[1000:]

    

    lr = ensemble.GradientBoostingRegressor() #Base Model

    lr.fit(X_train[cols], X_train[target].values.ravel())

    print ("Base Score: ", lr.score(X_train[cols], X_train[target].values.ravel())) # 0.9672992360887501

    best_score = 0

    best_cols = cols

    for k in range(len(cols)//4, len(cols)):

        lr = ensemble.GradientBoostingRegressor()

        curr_cols = select_cols(cols, target, X_train, k)

        lr.fit(X_train[curr_cols], X_train[target].values.ravel())

        os_score = lr.score(X_valid[curr_cols], X_valid[target].values.ravel())

        if os_score > best_score:

            is_score = lr.score(X_train[curr_cols], X_train[target].values.ravel())

            print ("K= ", k, ", IS score: ", is_score, ", OS score: ", os_score) # 0.840628507295174

            best_score = os_score

            best_cols = curr_cols

            

    return best_cols



best_cols = find_best_cols(cols, "SalePrice", train)

best_cols
print(len(best_cols))

print(len(cols))

cols = best_cols
train[cols].head()
# We will use this function to get the best model 

def get_tuned_model(estimator, param_grid, scoring, X_train, Y_train):

    from sklearn.model_selection import GridSearchCV



    grid = GridSearchCV(estimator = estimator, 

                       param_grid = param_grid,

                       scoring = scoring,

                       cv=3,

                       n_jobs= -1

                      )



    tuned = grid.fit(X_train, Y_train)



    print ("Best score: ", tuned.best_score_) # 0.840628507295174

    print ("Best params: ", tuned.best_params_)

    print ("IS Score: ", tuned.score(X_train, Y_train)) # 0.8698092031425365

    

    return tuned

param_grid = { 

    "learning_rate":  [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1],

    "n_estimators": [500]

}



gbc = ensemble.GradientBoostingRegressor()

gbc_tuned = get_tuned_model(gbc, param_grid, "r2", train[cols], train[["SalePrice"]].values.ravel())
# We will use this to save our results 

def save_results(model, data, ids):

    pred_test = model.predict(data) # Predict values for data

    test_res = ids.copy()

    test_res["SalePrice"] = pred_test

    test_res.to_csv("/kaggle/working/my_predictions.csv", index=False)

    return test_res
results = save_results(gbc_tuned, test[cols], test[['Id']])

results.head()
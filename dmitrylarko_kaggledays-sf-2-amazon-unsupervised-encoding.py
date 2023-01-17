import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



from catboost.datasets import amazon

train, test = amazon()

print(train.shape, test.shape)

target = "ACTION"

col4train = [x for x in train.columns if x not in [target, "ROLE_TITLE"]]

y = train[target].values
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate



# returns model instance

def get_model(): 

    params = {

        "n_estimators":300, 

        "n_jobs": 3,

        "random_state":5436,

    }

    return ExtraTreesClassifier(**params)



# validate model on given dataset and report CV score

def validate_model(model, data):

    skf = StratifiedKFold(n_splits=5, random_state = 4141, shuffle = True)

    stats = cross_validate(

        model, data[0], data[1], 

        groups=None, scoring='roc_auc', 

        cv=skf, n_jobs=None, return_train_score = True

    )

    stats = pd.DataFrame(stats)

    return stats.describe().transpose()



# transforms given train and test datasets using provided function, 

# function parameters can be passed as a dict

def transform_dataset(train, test, func, func_params = {}):

    dataset = pd.concat([train, test], ignore_index = True)

    dataset = func(dataset, **func_params)

    if isinstance(dataset, pd.DataFrame):

        new_train = dataset.iloc[:train.shape[0],:].reset_index(drop = True)

        new_test =  dataset.iloc[train.shape[0]:,:].reset_index(drop = True)

    else:

        new_train = dataset[:train.shape[0]]

        new_test =  dataset[train.shape[0]:]

    return new_train, new_test
MJTCP = 32292 #Michael Jordan total career points

#for each column in dataset creates N column with random integers

def assign_rnd_integer(dataset, number_of_times = 5, seed = MJTCP):

    new_dataset = pd.DataFrame()

    np.random.seed(seed)

    for c in dataset.columns:

        for i in range(number_of_times):

            col_name = c+"_"+str(i)

            unique_vals = dataset[c].unique()

            labels = np.array(list(range(len(unique_vals))))

            np.random.shuffle(labels)

            mapping = pd.DataFrame({c: unique_vals, col_name: labels})

            new_dataset[col_name] = (dataset[[c]]

                                     .merge(mapping, on = c, how = 'left')[col_name]

                                    ).values

    return new_dataset
new_train, new_test = transform_dataset(

    train[col4train], test[col4train], 

    assign_rnd_integer, {"number_of_times":5}

)

print(new_train.shape, new_test.shape)

new_train.head(5)
validate_model(

    model = get_model(), 

    data = [new_train.values, y]

)
new_train, new_test = transform_dataset(

    train[col4train], test[col4train], 

    assign_rnd_integer, {"number_of_times":1}

)

print(new_train.shape, new_test.shape)

validate_model(

    model = get_model(), 

    data = [new_train.values, y]

)
new_train, new_test = transform_dataset(

    train[col4train], test[col4train], 

    assign_rnd_integer, {"number_of_times":10}

)

print(new_train.shape, new_test.shape)

validate_model(

    model = get_model(), 

    data = [new_train.values, y]

)
from sklearn.preprocessing import OneHotEncoder

# transforms given dataset to OHE representation

def one_hot(dataset):

    ohe = OneHotEncoder(sparse=True, dtype=np.float32, handle_unknown='ignore')

    return ohe.fit_transform(dataset.values)
new_train, new_test = transform_dataset(

    train[col4train], test[col4train], 

    one_hot)

print(new_train.shape, new_test.shape)
#Warning!!! Long run, better skip it.

validate_model(

    model = get_model(), 

    data = [new_train, y]

)
from sklearn.decomposition import TruncatedSVD

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



def extract_col_interaction(dataset, col1, col2, tfidf = True):

    data = dataset.groupby([col1])[col2].agg(lambda x: " ".join(list([str(y) for y in x])))

    if tfidf:

        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(" "))

    else:

        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "))

    

    data_X = vectorizer.fit_transform(data)

    dim_red = TruncatedSVD(n_components=1, random_state = 5115)

    data_X = dim_red.fit_transform(data_X)

    

    result = pd.DataFrame()

    result[col1] = data.index.values

    result[col1+"_{}_svd".format(col2)] = data_X.ravel()

    return result



import itertools

def get_col_interactions_svd(dataset, tfidf = True):

    new_dataset = pd.DataFrame()

    for col1,col2 in itertools.permutations(dataset.columns, 2):

        data = extract_col_interaction(dataset, col1,col2, tfidf)

        col_name = [x for x in data.columns if "svd" in x][0]

        new_dataset[col_name] = dataset[[col1]].merge(data, on = col1, how = 'left')[col_name]

    return new_dataset
new_train, new_test = transform_dataset(

    train[col4train], test[col4train], 

    get_col_interactions_svd

)

print(new_train.shape, new_test.shape)

new_train.head(5)
validate_model(

    model = get_model(), 

    data = [new_train.values, y]

)
def get_freq_encoding(dataset):

    new_dataset = pd.DataFrame()

    for c in dataset.columns:

        data = dataset.groupby([c]).size().reset_index()

        new_dataset[c+"_freq"] = dataset[[c]].merge(data, on = c, how = "left")[0]

    return new_dataset
new_train, new_test = transform_dataset(

    train[col4train], test[col4train], 

    get_freq_encoding

)

print(new_train.shape, new_test.shape)

new_train.head(5)
validate_model(

    model = get_model(), 

    data = [new_train.values, y]

)
new_train1, new_test1 = transform_dataset(

    train[col4train], test[col4train], get_freq_encoding

)

new_train2, new_test2 = transform_dataset(

    train[col4train], test[col4train], get_col_interactions_svd

)

new_train3, new_test3 = transform_dataset(

    train[col4train], test[col4train], 

    assign_rnd_integer, {"number_of_times":10}

)



new_train = pd.concat([new_train1, new_train2, new_train3], axis = 1)

new_test = pd.concat([new_test1, new_test2, new_test3], axis = 1)

print(new_train.shape, new_test.shape)
validate_model(

    model = get_model(), 

    data = [new_train.values, y]

)
model = get_model()

model.fit(new_train.values, y)

predictions = model.predict_proba(new_test)[:,1]



submit = pd.DataFrame()

submit["Id"] = test["id"]

submit["ACTION"] = predictions



submit.to_csv("submission.csv", index = False)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
train_set = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test_set = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



train_set.head()
numeric_col_count = 0

total_col_count = 0

for column in train_set:

    # print whether this column is numeric or not

    is_numeric = pd.api.types.is_numeric_dtype(train_set[column])

    numeric_col_count += 1 if is_numeric else 0

    total_col_count += 1

    

    #print("Is " + column + " numeric? " + str(is_numeric))



print("Number of numeric columns: %i, number of non-numeric columns: %i" % (numeric_col_count, total_col_count - numeric_col_count))
train_m = len(train_set)

ohcs = []



# set up one hot encoders for the data

for column in train_set:

    # check whether this column is already numeric data

    is_numeric = pd.api.types.is_numeric_dtype(train_set[column])



    new_col = train_set[column].to_numpy().reshape((1, train_m))



    if not is_numeric:

        train_set[column] = train_set[column].astype(np.str).fillna(' ')



        # use one hot encoding to get rid of non-numeric data

        ohc = OneHotEncoder(handle_unknown='ignore')

        ohc.fit(train_set[column].to_numpy().reshape(-1, 1))

        

        ohcs.append(ohc)
# processes the data

def process_data(data):

    # list of data frames to concatenate later

    dfs = []

    nps = []

    

    # training data size

    m = len(data)

    processed = None

    

    # pick out id and remove it

    ids = data['Id'].to_numpy().reshape((1, m))

    data = data.drop(['Id'], axis = 1)

    

    # if there is sale price (it's not present in test set), pick it out and remove it

    price = []

    

    if 'SalePrice' in data:

        price = data['SalePrice'].to_numpy().reshape((1, m))

    

        # drop the id and sale price

        data = data.drop(['SalePrice'], axis = 1)

    

    # index for one-hot encoder

    ohc_i = 0

    

    for column in data:

        # check whether this column is already numeric data

        is_numeric = pd.api.types.is_numeric_dtype(data[column])

        

        new_col = data[column].to_numpy().reshape((1, m))

        

        if not is_numeric:

            data[column] = data[column].astype(np.str).fillna(' ')

            

            # use one hot encoding to get rid of non-numeric data

            ohc = ohcs[ohc_i] # OneHotEncoder()

            new_col = ohc.transform(data[column].to_numpy().reshape(-1, 1))

            new_col = new_col.toarray().T

            

            ohc_i += 1

        

        # concatenate

        if processed is not None:

            processed = np.concatenate((processed, new_col))

        else:

            processed = new_col

    

    # normalize

    #p_norm = np.linalg.norm(processed, axis=1, keepdims=True)

    #processed /= p_norm

    

    # features were concatenated on the 

    return processed, price, ids
train_X, train_Y, _ = process_data(train_set)

test_X, _, test_ids = process_data(test_set)



# sanity check

print("train_X size:  " + str(train_X.shape))

print("train_Y size:  " + str(train_Y.shape))

print("test_X size:   " + str(test_X.shape))

print("test_ids size: " + str(test_ids.shape))
# remove nans

train_X = np.nan_to_num(train_X)

test_X = np.nan_to_num(test_X)



# Split the model for scoring

X_train, X_test, Y_train, Y_test = train_test_split(train_X.T, train_Y.T, test_size=0.2)



# explore multiple random states

best_i = -1

best_score = 0

for i in range(20):

    # Set up and fit the model

    model = RandomForestRegressor(random_state=i)

    model.fit(X_train, Y_train.ravel())



    # Score the model

    score = model.score(X_test, Y_test.ravel())

    print("Score for %i: %.3f" % (i, score))

    

    # If it scored better than max, choose this

    if score > best_score:

        best_i = i

        best_score = score



# re-initialize and fit model based on the best i

model = RandomForestRegressor(n_estimators = 10, random_state=best_i)

model.fit(X_train, Y_train.ravel())



# Make some predictions

test_pred = model.predict(X_test[0:10, :])

print(test_pred)
# Predict all the values

predictions = model.predict(test_X.T)



# Format it like the example submission

output = {'Id': test_ids[0], 'SalePrice': predictions}

dataframe = pd.DataFrame(output, columns = ['Id', 'SalePrice'])



# Convert to CSV

dataframe.to_csv(r"./house_price_submission.csv", index=False, header=True)
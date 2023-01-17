import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
import lightgbm as lgb
def feature_engineering(df):
    df['Euclidean_Distance_To_Hydrology'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)
    df['Hillshade_Total'] = df.filter(like='Hillshade').sum(axis=1)
    df['Hillshade_Slope'] = (df['Hillshade_3pm'] - df['Hillshade_9am']) / df['Hillshade_9am'].apply(lambda x: 1 if x == 0 else x)
    return df

def collapse(df):
    wild = df.filter(like='Wilderness_Area', axis=1)
    soil = df.filter(like='Soil_Type', axis=1)

    df.drop(wild.columns, axis=1, inplace=True)
    df.drop(soil.columns, axis=1, inplace=True)

    for col in wild.columns:
        id = int(col[15:])
        wild.loc[:, col] = wild.loc[:, col] * id

    for col in soil.columns:
        id = int(col[9:])
        soil.loc[:, col] = soil.loc[:, col] * id

    df['Wilderness_Area'] = wild.sum(axis=1)
    df['Soil_Type'] = soil.sum(axis=1)

    return df

print('processing train data...')
df = pd.read_csv('../input/train.csv')
df = df.sample(frac=1) # Shuffling is necessary for the training set as the dataset is not shuffled.

train_df = feature_engineering(collapse(df))
print('\tdone.')

print('processing test data...')
df = pd.read_csv('../input/test.csv')

test_df = feature_engineering(collapse(df))
print('\tdone.')
def make_df_cat(df, cat_columns):
    for col in cat_columns:
        df[col] = pd.Categorical(df[col]).codes
    df[cat_columns] = df[cat_columns].astype('category')
    return df

def load_data(df):
    
    print('preparing data...')
    df = make_df_cat(df, cat_columns=['Wilderness_Area', 'Soil_Type', 'Cover_Type'])
    df.drop('Id', axis=1, inplace=True)

    print('\tbuilding test set...')
    X_test = df.iloc[:1512, :]
    y_test = X_test.pop('Cover_Type')
    df.drop(X_test.index, axis=0, inplace=True)

    print('\tbuilding training set...')

    X_train = df
    y_train = X_train.pop('Cover_Type')

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(train_df)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': ['multiclass', 'multi_error'],
    'num_class': 7,
    'learning_rate': 0.05,
    'verbose': 0,
    'num_boost_round': 1000,
    'num_leaves': 256,
    'max_depth': 128,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
}

print('training...')

gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)
print('predicting...')
id_col = test_df.pop('Id')

y = gbm.predict(make_df_cat(test_df, cat_columns=['Wilderness_Area', 'Soil_Type']))
y = [np.argmax(x)+1 for x in y]

submit = pd.DataFrame({'Id': id_col, 'Cover_Type': y})

submit.to_csv('submission.csv', index=False)
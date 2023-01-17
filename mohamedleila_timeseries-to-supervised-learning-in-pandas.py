import numpy as np

import pandas as pd
# example dataframe

columns = ['step', 'f1', 'f2', 'f3', 'f4']

data = [

    [1, 2, 3, 4, 5],

    [2, 7, 8, 9, 3],

    [3, 4, 5, 0, 0],

    [4, 11, 12, 9, 2],

    [5, 12, 12, 8, 3],

    [6, 1, 17, 9, 6],

    [7, 16, 62, 2, 2],

    

    [1, 2, 3, 4, 5],

    [2, 7, 8, 9, 3],

    [3, 4, 5, 0, 0],

    [4, 11, 12, 9, 2],

    [5, 12, 12, 8, 3],

    [6, 1, 17, 9, 6],

    [7, 16, 62, 2, 2]

]

df = pd.DataFrame(data, columns=columns)

df
# function

def transform(df, step, features, targets, lags=4):

    def combine_cols(df, features):

        df['col_list'] = df[features].values.tolist()

        return df



    target_indices = [idx for idx, val in enumerate(features) if val in targets]

    

    df = df.groupby(step).apply(combine_cols, features)

    

    lag_cols = []

    for lag in reversed(range(0, lags+1)):

        col_name = "lag_" + str(lag)

        lag_cols.append(col_name)

        df[col_name] = df["col_list"].shift(lag)



    df = df.drop(features+["col_list"], axis=1).dropna()

    

    lag_cols = [col for col in df.columns if 'lag_'in col]

    df['concat_feats'] = np.empty((len(df), 0)).tolist()

    df['concat_targets'] = np.empty((len(df), 0)).tolist()

    

    last_lag = len(lag_cols)

    for idx, col in enumerate(lag_cols):

        if idx == last_lag - 1:

            # extarct targets

            df['concat_targets'] = df[col].apply(lambda x: np.take(x, indices=target_indices))

            df.drop(col, axis=1, inplace=True)

            continue

        df['concat_feats'] += df[col]

        df.drop(col, axis=1, inplace=True)

    

    return df
# result

transform(df, step=['step'], features=['f1', 'f2', 'f3', 'f4'], targets=['f2', 'f3'], lags=3)
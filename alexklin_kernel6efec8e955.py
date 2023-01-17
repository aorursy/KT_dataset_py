import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from lightgbm import LGBMRegressor

from sklearn import metrics
# Functions



def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.   

        https://www.kaggle.com/gemartin/load-data-reduce-memory-usage 

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))    

    for col in df.columns:

        col_type = df[col].dtype        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))    

    return df





def metrics_regression(true, predicted):  

    """Metrics"""

    r2_square = metrics.r2_score(true, predicted) 

    print(f'R2_predicted:     {r2_square:.3f}')

    



def feature_importances(x, model):

    df = pd.DataFrame(sorted(list(zip(x, model.feature_importances_)), 

                      key=lambda x: x[1]))

    plt.figure(figsize=(12, 7))

    plt.barh(df[0], df[1])

    plt.show()

    

    

def model_LGBMRegressor(X, y):

    model = LGBMRegressor(max_depth=7,

                          min_samples_leaf=10,

                          n_estimators=300, 

                          random_state=42,

                          n_jobs=-1)

    model.fit(X, y)

    cv_score = cross_val_score(model, X, y, 

                               scoring='r2', 

                               cv=KFold(n_splits=5, shuffle=True, random_state=42))

    # cv_score

    mean = cv_score.mean()

    std = cv_score.std()

    print()

    print('CV_score R2 mean: {:.3f} +- {:.3f}'.format(mean, std))    

    predictions = model.predict(X)

    metrics_regression(y, predictions)

    feature_importances(X, model)

    return model
train = pd.read_csv('/kaggle/input/realestatepriceprediction/train.csv')

y = train['Price']

X = train.drop('Price',  axis=1)



# Feature engineering

def feature_engineering(X):

    X = X.drop(['Id'],  axis=1)

    X['Healthcare_1'].fillna(X['Healthcare_1'].median(), inplace=True)

    X['LifeSquare'].fillna(X['Square'] * 0.7, inplace=True)

    X = reduce_mem_usage(X)

    print('Shape:', X.shape)

    return X



# Model

X = feature_engineering(X)

model = model_LGBMRegressor(X, y)
# Final



X_final = pd.read_csv('/kaggle/input/realestatepriceprediction/test.csv')

preds_final = pd.DataFrame()

preds_final['Id'] = X_final['Id'].copy()



X_final = feature_engineering(X_final)

preds_final['Price'] = model.predict(X_final)

preds_final.to_csv('final.csv', index=False)

preds_final.head()
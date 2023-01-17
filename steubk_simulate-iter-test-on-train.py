import numpy as np 

import pandas as pd 
def simulate_iter_test_on_train_data (num_plays = 3438):

    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv',  low_memory=False)

    train = train.drop ( ["Yards"], axis=1 )

    steps = min (num_plays, train.shape[0] // 22)

    cols = ["Yards"+str(x) for x in range (-99,100)] 

    

    pred_data = np.zeros ( (steps, 199)) 

    train_pred = pd.DataFrame ( pred_data, columns = cols  )

    

    for i in range (steps):

        first_row =i*22 

        df = train[first_row:first_row + 22]

        sub = train_pred [i:i+1]

        yield df, sub



%%time



for n, (simulated_test,sample_prediction) in enumerate(simulate_iter_test_on_train_data(num_plays=3438)):

    # code to test

    simulated_test = simulated_test



print(n)
%%time 

from kaggle.competitions import nflrush

env = nflrush.make_env()

for n,(test, test_prediction) in enumerate(env.iter_test()):

    # code to run

    test=test

    env.predict(test_prediction)

print(n)

 
# return the dataframe with the distance to running back (apply method)    

def process_apply ( df ):

    def euclidean_distance(x1,y1,x2,y2):

        x_diff = (x1-x2)**2

        y_diff = (y1-y2)**2

        return np.sqrt(x_diff + y_diff)

    

    carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y']]

    df = df.merge ( carriers.rename(columns={'X':'back_X', 'Y':'back_Y'}), on=["GameId","PlayId"], how='inner' ) 

    df['dist_to_back'] = df[['X','Y','back_X','back_Y']].apply(lambda x: euclidean_distance(x[0],x[1],x[2],x[3]), axis=1)

    

    return df



# return the dataframe with the distance to running back (vectorization)  

def process_vectorized ( df ):

    

    carriers = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y']]

    df = df.merge ( carriers.rename(columns={'X':'back_X', 'Y':'back_Y'}), on=["GameId","PlayId"], how='inner' ) 

    df['dist_to_back'] = ((df['X'] - df["back_X"])**2 + (df['Y'] - df["back_Y"])**2)**(1/2) 

    

    return df





# return the dataframe with the distance to running back (vectorization in case of dataframe with a single play)  

def process_vectorized_single_play ( df ):

    

    

    running_back = df[df['NflId'] == df['NflIdRusher']][['GameId','PlayId','X','Y']]

    # no need to merge



    back_X = running_back['X'].values[0]

    back_Y = running_back['Y'].values[0]



    df['dist_to_back'] = ((df['X'] - back_X )**2 + (df['Y'] - back_Y)**2)**(1/2) 

    

    return df





train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv',  low_memory=False)
%%time

train_apply = process_apply(train)
%%time

train_vectorized = process_vectorized(train)
from pandas.testing import assert_series_equal

assert_series_equal (train_apply['dist_to_back'], train_vectorized['dist_to_back'] )
%%time 

result = np.zeros ( (3438*22,) ) 

for n, (simulated_test,sample_prediction) in enumerate(simulate_iter_test_on_train_data(num_plays=3438)):

    simulated_test = process_apply (simulated_test)

    result[n*22:(n+1)*22] = simulated_test["dist_to_back"].values



result_apply = pd.Series(result)
%%time 

result = np.zeros ( (3438*22,) ) 

for n, (simulated_test,sample_prediction) in enumerate(simulate_iter_test_on_train_data(num_plays=3438)):

    simulated_test = process_vectorized (simulated_test)

    result[n*22:(n+1)*22] = simulated_test["dist_to_back"].values



result_vectorized = pd.Series(result)
%%time 

result = np.zeros ( (3438*22,) ) 

for n, (simulated_test,sample_prediction) in enumerate(simulate_iter_test_on_train_data(num_plays=3438)):

    simulated_test = process_vectorized_single_play (simulated_test)    

    result[n*22:(n+1)*22] = simulated_test["dist_to_back"].values



result_vectorized_single_play = pd.Series(result)
assert_series_equal (result_apply, result_vectorized)

assert_series_equal (result_vectorized, result_vectorized_single_play)
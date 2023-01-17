import numpy as np 

import pandas as pd
def resize_data(dataset): 

    dataset.replace(' NA', -99, inplace=True)

    dataset.fillna(-99, inplace=True)

    

    for col in list(dataset.columns):

        if dataset[col].dtype == 'int64' or dataset[col].dtype == 'float64':

            dataset[col] = dataset[col].astype(np.int8)    

                

    return dataset
reader = pd.read_csv('../input/train_ver2.csv', chunksize=10000)

df = pd.concat([resize_data(chunk) for chunk in reader])
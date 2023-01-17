import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def data_loader():
    global df_train_all, df_test_all, df_sub_all
    print('Loading data ...')
    df_train_all = pd.read_csv('../input/train.csv', dtype={'x':np.float32, 
                                               'y':np.float32, 
                                               'accuracy':np.int16,
                                               'time':np.int,
                                               'place_id':np.int}, 
                                               index_col = 0)
    df_test_all = pd.read_csv('../input/test.csv', dtype={'x':np.float32,
                                              'y':np.float32, 
                                              'accuracy':np.int16,
                                              'time':np.int,
                                              'place_id':np.int}, 
                                              index_col = 0)
    df_sub_all = pd.read_csv('../input/sample_submission.csv', index_col = 0)
    print('data loaded')
    
data_loader()
df_train_sample = df_train_all.sample(frac=0.1)
df_train_sample = df_train_sample.sort_values(by=['time'])
validation_ratio = 0.1
train_size = int((1.0 - validation_ratio) * len(df_train_sample))
sample_train = df_train_sample[:train_size]
sample_validation = df_train_sample[train_size:]
place_ids, place_id_counts = np.unique(sample_train['place_id'].values, return_counts=True)
place_id_counts_combine = np.asarray((place_ids, place_id_counts)).T
res = np.sort(place_id_counts_combine, 0)
print(res[:50])

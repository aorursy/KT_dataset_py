import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
df_train = pd.read_csv('../input/train.csv',
                       usecols=['row_id','x','y','time','place_id', 'accuracy'], 
                       index_col = 0)
df_test = pd.read_csv('../input/test.csv',
                      usecols=['row_id','x','y','time', 'accuracy'],
                      index_col = 0)
print('Data loaded.')
h_round_base = 0.5 #We round hours to nearest h_round_base

def prepare_data(df, n_cell_x, n_cell_y):
    """
    Feature engineering and computation of the grid.
    """
    #Creating the grid
    size_x = 10. / n_cell_x
    size_y = 10. / n_cell_y
    eps = 0.00001  
    xs = np.where(df.x.values < eps, 0, df.x.values - eps)
    ys = np.where(df.y.values < eps, 0, df.y.values - eps)
    pos_x = (xs / size_x).astype(np.int)
    pos_y = (ys / size_y).astype(np.int)
    df['grid_cell'] = pos_y * n_cell_x + pos_x
    
    #Feature engineering
    fw = [500, 1350, 3, 2, 1./40., 2, 10, 1./27.] #feature weights (black magic here)
    min_max_scaler = preprocessing.MinMaxScaler()

    df.x = df.x.values * fw[0]
    df.y = df.y.values * fw[1]

    initial_date = np.datetime64('2016-02-01T01:01', dtype='datetime64[m]') 
    d_times = pd.DatetimeIndex(initial_date + np.timedelta64(int(mn), 'm') 
                               for mn in df.time.values)  
    d_hour = d_times.hour + (d_times.minute / 60.)
    df['hour'] = ( (d_hour / h_round_base).round() * h_round_base) * fw[2]
    df['weekday'] = d_times.weekday * fw[3]
    df['day'] = (d_times.day * fw[4])
    df['month'] = d_times.month * fw[5]
    df['year'] = (d_times.year - 2013) * fw[6]
    
    df.accuracy = df.accuracy.values * fw[7]
    
    df = df.drop(['time'], axis=1) 
    return df
def process_one_cell(df_train, df_test, grid_id, th):
    """   
    Classification inside one grid cell.
    """   
    #Working on df_train
    df_cell_train = df_train.loc[df_train.grid_cell == grid_id]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    row_ids = df_cell_test.index
    
    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id', 'grid_cell'], axis=1).values.astype(int)
    X_test = df_cell_test.drop(['grid_cell'], axis = 1).values.astype(int)
    
    #Applying the classifier
    clf = KNeighborsClassifier(n_neighbors=20, weights='distance', 
                               metric='manhattan')

    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])    
    return pred_labels, row_ids
def process_grid(df_train, df_test, th, n_cells):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """ 
    preds = np.zeros((df_test.shape[0], 3), dtype=int)
    
    for g_id in range(n_cells):
        #Applying classifier to one grid cell
        pred_labels, row_ids = process_one_cell(df_train, df_test, g_id, th)

        #Updating predictions
        preds[row_ids] = pred_labels
        
        print('\rProcessed ' + str(g_id + 1) + ' / ' + str(n_cells) + ' cells.', end='')

    return preds
#Defining the size of the grid
#n_cell_x = 10
#n_cell_y = 10

#print('Preparing train data')
#df_train = df_train[df_train.accuracy < 1000]

#df_train = prepare_data(df_train, n_cell_x, n_cell_y)

#print('Preparing test data')
#df_test = prepare_data(df_test, n_cell_x, n_cell_y)

#Solving classification problems inside each grid cell
#th = 1 #Keeping place_ids with more than th samples. 
#preds = process_grid(df_train, df_test, th, n_cell_x*n_cell_y)

#Auxiliary dataframe with the 3 best predictions for each sample
#df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])  
    
#Concatenating the 3 predictions for each sample
#ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')
    
#Writting to csv
#ds_sub.name = 'place_id'
#ds_sub.to_csv('fb_kaggle_submission.csv', index=True, header=True, index_label='row_id') 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from pandas import read_csv
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from numpy import concatenate
from sklearn.neighbors import KNeighborsClassifier
import time
import os
def l2_dist(p1, p2):
    x1,y1 = p1
    x2,y2 = p2
    x1, y1 = np.array(x1), np.array(y1)
    x2, y2 = np.array(x2), np.array(y2)
    dx = x1 - x2
    dy = y1 - y2
    dx = dx ** 2
    dy = dy ** 2
    dists = dx + dy
    dists = np.sqrt(dists)
    return np.mean(dists), dists
def binarize(valArr):
    '''Return a binarized vector for each RSSI value'''
    low, high = -50, -200
    interval = 10
    bins = []
    for val in valArr:
        numBins = int(-(high - low)/interval)
        bin1 = np.zeros(numBins)
        idx = int((val - low) / interval)
        bin1[idx] = 1
        bins.append(bin1)
    return np.array(bins)
def loadData(path='../input/iBeacon_RSSI_Labeled.csv'):
    x = []
    le_col = LabelEncoder()
    le_row = LabelEncoder()
    x = read_csv(path, index_col=None)
    binDf = DataFrame()
    cols = ['b' + str(i+ 3000) for i in range(1, 14)]
    bined = []

    for col in cols:
        dt = binarize(x[col])
        for i in range(dt.shape[1]):
            binDf[col +'_'+ str(i)] = dt[:, i]
    # Seperate x and y coordinates.
    x['col_loc'] = x['location'].str[0]
    x['row_loc'] = x['location'].str[1:]
    # transform coordinates to start from 0.
    binDf['x'] = le_col.fit_transform(x['col_loc'])
    binDf['y'] = le_row.fit_transform(x['row_loc'])

    return binDf 
def create_deep(inp_dim, num_classes):
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Dense(inp_dim, input_dim=inp_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
def main():
    df = loadData()
    print(df.head(3))
    
    '''
    Localizations prediction
    Here we develop two seperate models each of which is traind on the training data and predict one coordinate.
    '''
    
    bcol = df.columns[:-2] # keep features except the class labels.
    bcol2 = list(bcol)
    bcol2.extend(['x'])
    bcol3 = list(bcol)
    bcol3.extend(['y'])
    df_ = df 
    parameters = df_.values.shape[1] - 1 
    values = df_.values
    
    df_ = df_.sample(frac=1).reset_index(drop=True) # shuffle data
    df2 = DataFrame(df_[bcol2]) # dataframe for training model for X coordinate
    df3 = DataFrame(df_[bcol3]) # dataframe for training model for Y coordinate

    # Split dataset for train and test.
    lenx = int(len(values)*.7)
    ## predict x positions 
    values2 = df2.values
    train2 = values2[:lenx, :]
    test2  = values2[lenx:, :]

    train2_X, train2_y = train2[:, :-1], train2[:, -1]
    test2_X, test2_y = test2[:, :-1], test2[:, -1]

    train2_yc = np_utils.to_categorical(train2_y)
    test2_yc = np_utils.to_categorical(test2_y)
    numclass = train2_yc.shape[1]

    model = create_deep(parameters-1, numclass)
    model.fit(train2_X,train2_yc, epochs=100, batch_size=10,  verbose=0)

    # calculate predictions
    predictions2 = model.predict(test2_X)
    Xs_pred = np.argmax(predictions2, axis=1)
    Xs_orig = np.argmax(test2_yc, axis=1)
    
    ## predict y positions 
    values3 = df3.values
    train3 = values3[:lenx, :]
    test3  = values3[lenx:, :]

    train3_X, train3_y = train3[:, :-1], train3[:, -1]
    test3_X, test3_y = test3[:, :-1], test3[:, -1]

    train3_yc = np_utils.to_categorical(train3_y)
    test3_yc = np_utils.to_categorical(test3_y)
    numclass = train3_yc.shape[1]

    model2 = create_deep(parameters-1, numclass)
    model2.fit(train3_X,train3_yc, epochs=100, batch_size=10,  verbose=2)

    # calculate predictions
    predictions3 = model2.predict(test3_X)

    Ys_pred = np.argmax(predictions3, axis=1)
    Ys_orig = np.argmax(test3_yc, axis=1)
    
    #compute the Euclidean distance error
    l2dists_mean, l2dists = l2_dist((Xs_orig, Ys_orig), (Xs_pred, Ys_pred))
    print ('L2 error:', l2dists_mean)
    
    # Prepare for ploting Cumulative Density Function (CDF) of errors
    sortedl2_deep = np.sort(l2dists)
    prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)
    fig, ax = plt.subplots()
    lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')
    plt.title('CDF of Euclidean distance error')
    plt.xlabel('Distance (m)')
    plt.ylabel('Probability')
    plt.grid(True)
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('-.')

    plt.savefig('Figure_CDF_error.png', dpi=300)
    plt.show()
    plt.close()

main()

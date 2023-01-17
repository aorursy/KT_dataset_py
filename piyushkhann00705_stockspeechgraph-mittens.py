# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np 

import pandas as pd

import pickle

import os

import datetime

from tqdm import tqdm

from statistics import mean 

import matplotlib.pyplot as plt 

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation , Masking, Bidirectional, TimeDistributed, Input,concatenate

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import EarlyStopping

from keras.models import Sequential

import scipy.stats as stats

from scipy.stats import pearsonr

from scipy.stats import spearmanr

from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV



duration_list=[]

batch_sizes=[]

epochs_list=[]

optimizer_list=[]

training_loss_list=[]

test_loss_list=[]

pearson_list=[]

spearman_list=[]



    

traindf= pd.read_csv("../input/stockspeechgraph/train_split2.csv")

testdf=pd.read_csv("../input/stockspeechgraph/test_split2.csv")

valdf=pd.read_csv("../input/stockspeechgraph/val_split2.csv")



train_and_val_df= pd.read_csv("../input/stockspeechgraph/train_and_val.csv")

    

with open('../input/stockspeechgraph/mittens_train.pkl', 'rb') as f:

    text_train=pickle.load(f)



with open('../input/stockspeechgraph/mittens_train_and_val.pkl', 'rb') as f:

    text_train_and_val=pickle.load(f)

    

with open('../input/stockspeechgraph/mittens_test.pkl', 'rb') as f:

    text_test=pickle.load(f)

    

with open('../input/stockspeechgraph/mittens_val.pkl', 'rb') as f:

    text_val=pickle.load(f)

    

    

with open('../input/stockspeechgraph/earnings_wiki_knowgraph.pickle', 'rb') as f:

    graph_dict=pickle.load(f)

    

    

error=[]





def ModifyData(df,text_dict):

    X=[]

    y_3days=[]

    y_7days=[]

    y_15days=[]

    y_30days=[]



    for index,row in df.iterrows():

        



        

        try:

            embed=text_dict[row['text_file_name'][:-9]]

            graph_embed=graph_dict[row['text_file_name'][:-18]]

            embed=np.concatenate([graph_embed,embed])

            X.append(embed)

            y_3days.append(float(row['future_3']))

            y_7days.append(float(row['future_7']))

            y_15days.append(float(row['future_15']))

            y_30days.append(float(row['future_30']))

        except:

            error.append(row['text_file_name'][:-9])





        

    X=np.array(X)

    y_3days=np.array(y_3days)

    y_7days=np.array(y_7days)

    y_15days=np.array(y_15days)

    y_30days=np.array(y_30days)

    

        

    return X,y_3days,y_7days,y_15days,y_30days







X_train_and_val,y_train_and_val3days, y_train_and_val7days, y_train_and_val15days, y_train_and_val30days=ModifyData(train_and_val_df,text_train_and_val)



X_test, y_test3days, y_test7days, y_test15days, y_test30days=ModifyData(testdf,text_test)



# X_val,y_val3days, y_val7days, y_val15days, y_val30days=ModifyData(valdf,text_val)





# input_text_shape = (X_train_text.shape[1],X_train_text.shape[2])

print(X_train_and_val.shape)
error


def SVR_model(duration,X_train, y_train, X_test, y_test):

    Cs = [0.001, 0.01, 0.1, 1, 10]

    gammas = [0.001, 0.01, 0.1, 1]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid)

    grid_search.fit(X_train, y_train)

    pred=grid_search.predict(X_test)

    

    

    save_path="SVR_predictions_{}.pkl".format(duration)

    

    pickle.dump(pred, open(save_path, 'wb'))

    

    # StatMetrics(pred,y_test)

    print("Duration="+str(duration))

    print("MSE:"+str(mean_squared_error(pred,y_test)))

    

    # grid_search.best_params_

    # return grid_search.best_params_

    return
SVR_model(3,X_train_and_val,y_train_and_val3days,X_test,y_test3days)
SVR_model(7,X_train_and_val,y_train_and_val7days,X_test,y_test7days)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime 

from sklearn.linear_model import LinearRegression
Data = pd.read_csv("/kaggle/input/sputnik/train.csv")

Data['error']  = np.linalg.norm(Data[['x', 'y', 'z']].values - Data[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

Data['epoch'] = Data['epoch'].apply(str).astype(str).apply(datetime.datetime.strptime, args= (["%Y-%m-%dT%H:%M:%S.%f"]))

Data.info()

def feature_gen(data, list_of_features, lag_len = 24):

    for i in list_of_features:

        for j in range(1,lag_len,1):

            data[i+"_lag_" + str(j)] = data[[i]].shift(j)

    

    return data



def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
def predict_err(data):

    

   

     

    data_t = feature_gen(data,

    [

     'x_sim',

     'y_sim',

     'z_sim' ] , lag_len = 10)

    

    features = [i for i in data_t.columns if 'sim' in i ]

    



   

    

    data_train = data_t[data_t.type == "train"].dropna()

    data_predict = data_t[data_t.type == 'test']

    

    

    X_x_t, Y_x_t= data_train[features] , data_train[['x']]

    X_x_p= data_predict[features] 

   

    X_y_t, Y_y_t= data_train[features] , data_train[['y']]

    X_y_p= data_predict[features] 

    



    X_z_t, Y_z_t= data_train[features] , data_train[['z']]

    X_z_p= data_predict[features] 

    

    

    

    model_x = LinearRegression()

    model_y = LinearRegression()

    model_z = LinearRegression()

    

    

    model_x.fit(X_x_t, Y_x_t)

    model_y.fit(X_y_t, Y_y_t)

    model_z.fit(X_z_t, Y_z_t)

    

    

    data_predict['x'] = model_x.predict(X_x_p)

    data_predict['y'] = model_y.predict(X_y_p)

    data_predict['z'] = model_z.predict(X_z_p)

    

    data_predict['error'] = np.linalg.norm(data_predict[['x', 'y', 'z']].values - data_predict[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

    

    

    

    

   

    

    return data_predict[['id', 'error']]

    

    

    
gen_res = pd.DataFrame([], columns = ['id', 'error'])

for i in Data.sat_id.unique():

    

    gen_res = gen_res.append(predict_err(Data[Data.sat_id == i]))
gen_res
gen_res.to_csv("msub.csv",index= False)
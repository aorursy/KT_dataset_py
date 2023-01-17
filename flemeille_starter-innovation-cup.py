# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/ID_Data_train.csv')

test = pd.read_csv('../input/ID_Data_test.csv')

id_train = pd.read_csv('../input/ID_Time_train.csv')

sample_sub = pd.read_csv('../input/sample_sub.csv')
Ids_train = id_train['id_bateau_hash'].values

Ids_test = test['id_bateau_hash'].unique()

print(len(Ids_train), len(Ids_test))
train.head()
def create_df_model(input_data, IDs ,time_id=None, is_test=False):

    df_model = pd.DataFrame()

    data = input_data.copy()

    print('Calcul pour', len(IDs), 'individus')



    print(data.shape)

    for i in IDs:

        data_id = data[data['id_bateau_hash']==i]



        if is_test==False:

            df_model.loc[i, 'target'] = time_id[time_id['id_bateau_hash']==i]['Time'].values



        

        

        

        df_model.loc[i, 'longitude_std'] = data_id['longitude'].std()

        df_model.loc[i, 'latitude_mean'] = data_id['longitude'].mean()

        

    df_model = df_model.replace([np.inf, -np.inf], np.nan)

    df_model = df_model.fillna(0)

    

    print('finish !')

    return df_model
df_model_train = create_df_model(train, id_train['id_bateau_hash'], id_train)

df_model_test = create_df_model(test, Ids_test, is_test=True)
df_model_train.head()
df_model_test.head()
def rmse(y, y_pred):

    return np.sqrt(np.mean(np.square(y - y_pred)))
X_train = df_model_train.drop('target', axis=1)

y_train = df_model_train['target']

X_test = df_model_test.copy()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_predict

LR = LinearRegression()

y_preds_LR = cross_val_predict(LR, X_train, y_train, cv=3, verbose=10)

print(rmse(y_train, y_preds_LR))
LR.fit(X_train, y_train)

y_test_preds_lr = LR.predict(X_test)



sub = sample_sub 

sub['Time'] = y_test_preds_lr



sub.to_csv("my_awesome_pred.csv")
sub.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
limit_rows = 7000000;

df_train_y = pd.read_csv("../input/train_ver2.csv", usecols = [0,1] + list(range(24,45)), nrows=limit_rows);

# TODO: add last 3 cases
number_of_accounts = 21;

df_train_y.iloc[:,list(range(2,23))] = df_train_y.iloc[:,list(range(2,23))].astype('int8');

print(df_train_y.dtypes);
# get unique insert date

unique_fecha_dato = df_train_y.fecha_dato.unique();

length_unique_fecha_dato = len(set(unique_fecha_dato));

print(unique_fecha_dato);

print("Number of unique dates in train : ", length_unique_fecha_dato);
# how many people opened an new account between the first two month

df_first_month = df_train_y.loc[df_train_y['fecha_dato']=='2015-01-28'];



# get unique user id

unique_ncodpers = df_first_month.ncodpers.unique();

print("Number of customers in train : ", len(set(unique_ncodpers)));



#for user_id in unique_ncodpers:

    
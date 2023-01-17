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
import random

def random_sampler(filename, k):
    sample = []
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        filesize = f.tell()
        
        random_set = sorted(random.sample(range(filesize), k))
       
        for i in range(k):
            f.seek(random_set[i])
            # Skip current line (because we might be in the middle of a line) 
            f.readline()
            # Append the next line to the sample set 
            sample.append(f.readline().rstrip())

    return sample
TRAIN_SAMPLES = 5*10**6
train_sample = random_sampler('../input/train.csv', TRAIN_SAMPLES)
train_sample[0].decode().split(',')
train_sample_ = [row.decode().split(",") for row in train_sample]
train = pd.DataFrame(train_sample_)
del train_sample
del train_sample_
train_df = pd.read_csv('../input/train.csv', nrows=1)
train.columns = train_df.columns
train = train.apply(pd.to_numeric)
train.info()
train['Demanda_uni_equil'] = np.log1p(train['Demanda_uni_equil'])
x_cols = train.columns
x_cols = x_cols.drop(['Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima', 'Demanda_uni_equil'])
print(x_cols)
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(train[x_cols], train['Demanda_uni_equil'])
test = pd.read_csv('../input/test.csv')
test.info()
test['Demanda_uni_equil'] = np.expm1(model.predict(test[x_cols]))
test[['id', 'Demanda_uni_equil']].to_csv('predictions_rf_random_sampling.csv', index=False)
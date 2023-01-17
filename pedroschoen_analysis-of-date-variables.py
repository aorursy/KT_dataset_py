import numpy
import pandas 

import os
print(os.listdir("../input"))


import matplotlib.pyplot as plt
%matplotlib inline

train = pandas.read_csv('../input/train.csv')
test = pandas.read_csv('../input/test.csv')
for i in [train,test]:
    i['activation_date'] = pandas.to_datetime(i['activation_date'])
    i['month'] = i['activation_date'].dt.month
    i['day'] = i['activation_date'].dt.day
    i['day_of_week'] = i['activation_date'].dt.dayofweek
    i['weekofyear'] = i['activation_date'].dt.weekofyear
### Thanks SRK for this script for venn plots https://www.kaggle.com/sudalairajkumar/simple-exploration-baseline-notebook-avito

from matplotlib_venn import venn2

for i in ['month','day','day_of_week','weekofyear']:

    plt.figure(figsize=(10,7))
    venn2([set(train[i].unique()), set(test[i].unique())], set_labels = ('Train set', 'Test set') )
    plt.title("Number of " + i + " in train and test", fontsize=15)
    plt.show()


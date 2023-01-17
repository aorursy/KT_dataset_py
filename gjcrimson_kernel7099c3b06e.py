# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

np.random.seed(0)



import matplotlib.pyplot as plt



from sklearn import preprocessing

from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv( '../input/pubg-finish-placement-prediction/train_V2.csv')



df_test = pd.read_csv( '../input/pubg-finish-placement-prediction/test_V2.csv')
df_train['matchType'] = df_train['matchType'].map({

    'crashfpp':2,

    'crashtpp':2,

    'flarefpp':4,

    'flaretpp':4,

    'solo':1,

    'solo-fpp':1,

    'normal-solo':1,

    'normal-solo-fpp':1,   

    'duo':2,

    'duo-fpp':2,

    'normal-duo':2,

    'normal-duo-fpp':2,  

    'squad':4,

    'squad-fpp':4,

    'normal-squad':4,

    'normal-squad-fpp':4

}

)

df_test['matchType'] = df_test['matchType'].map({

    'crashfpp':2,

    'crashtpp':2,

    'flarefpp':4,

    'flaretpp':4,

    'solo':1,

    'solo-fpp':1,

    'normal-solo':1,

    'normal-solo-fpp':1,   

    'duo':2,

    'duo-fpp':2,

    'normal-duo':2,

    'normal-duo-fpp':2,  

    'squad':4,

    'squad-fpp':4,

    'normal-squad':4,

    'normal-squad-fpp':4

}

)

df_train
target = 'winPlacePerc'

features = list(df_train.columns)

features.remove("Id")

features.remove("matchId")

features.remove("groupId")
df_train = df_train.dropna(axis=0,how='any')

df_train
y_train = np.array(df_train[target])

features.remove(target)

x_train = df_train[features]

x_test = df_test[features]

x_train

random_seed=1

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
RFR = RandomForestRegressor(n_estimators=60, min_samples_leaf=3, max_features=0.5, n_jobs=-1)
%%time

RFR.fit(x_train, y_train)
print('mae train: ', mean_absolute_error(RFR.predict(x_train), y_train))

print('mae val: ', mean_absolute_error(RFR.predict(x_val), y_val))
pred = RFR.predict(x_test)

df_test['winPlacePerc'] = pred

submission = df_test[['Id', 'winPlacePerc']]

submission.to_csv('submission.csv', index=False)
pd.DataFrame({'features':x_train.columns,'importance':RFR.feature_importances_}).sort_values('importance',ascending=False)
plt.hist(pred)

plt.hist(y_train)
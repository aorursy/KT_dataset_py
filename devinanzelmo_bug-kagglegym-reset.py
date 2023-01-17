import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# code from SRK's kernel https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/univariate-model

import kagglegym

from sklearn import linear_model as lm





# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# Get the train dataframe

train = observation.train

mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)



cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']





models_dict = {}

for col in cols_to_use:

    model = lm.LinearRegression()

    model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)

    models_dict[col] = model



col = 'technical_20'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))



    observation, reward, done, info = env.step(target)

    if done:

        break

    

info





# We get our initial observation by calling "reset"

observation = env.reset()



# Get the train dataframe

train = observation.train

mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)



cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']





models_dict = {}

for col in cols_to_use:

    model = lm.LinearRegression()

    model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)

    models_dict[col] = model



col = 'technical_30'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))



    observation, reward, done, info = env.step(target)

    if done:

        break

    

info

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# Get the train dataframe

train = observation.train

mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)



cols_to_use = ['technical_30', 'technical_20', 'fundamental_11', 'technical_19']





models_dict = {}

for col in cols_to_use:

    model = lm.LinearRegression()

    model.fit(np.array(train[col].values).reshape(-1,1), train.y.values)

    models_dict[col] = model



col = 'technical_30'

model = models_dict[col]

while True:

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[col].values).reshape(-1,1)

    observation.target.y = model.predict(test_x)

    #observation.target.fillna(0, inplace=True)

    target = observation.target

    timestamp = observation.features["timestamp"][0]

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))



    observation, reward, done, info = env.step(target)

    if done:

        break

    

info

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
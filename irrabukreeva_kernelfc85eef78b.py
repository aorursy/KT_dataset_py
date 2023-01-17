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
import warnings

warnings.filterwarnings('ignore')



from sklearn.linear_model import LinearRegression
initi = pd.read_csv('/kaggle/input/sputnik/train.csv')
lag_period = 24

resultsx = pd.DataFrame()



for j in range(600):

    dfr = initi[initi.sat_id==j][['x','type']]

    train = dfr[dfr.type=='train'].reset_index(drop=True)

    test = dfr[dfr.type=='test'].reset_index(drop=True)

    

    i=0

    

    while(i*24<len(test)):

        df = pd.concat((train, test), axis = 0, ignore_index=True)

        df['target'] = df.x

        features = []

        for period_mult in range(1,3,1):

            df["lag_period_{}".format(period_mult)] = df.target.shift(period_mult*lag_period)

            features.append("lag_period_{}".format(period_mult))

        df['lagf_mean'] = df[features].mean(axis = 1)

        features.extend(['lagf_mean'])

            

        model = LinearRegression()



        train_df = df[df.type=='train'][features + ['target']].dropna()

    

        test_df = df[df.type=='test'][features].reset_index(drop=True)

        test_df = test_df.loc[:23,:]

        

        model.fit(train_df.drop(['target'], axis = 1), train_df['target'])

        forecast = model.predict(test_df)

        test.loc[i*24:i*24+23,:]['x'] = forecast

        test.loc[i*24:i*24+23,:]['type'] = 'train'



        i+=1

    

    resultsx = pd.concat((resultsx, test),axis=0, ignore_index=True)
resultsx.drop(['type'], axis=1, inplace=True)
lag_period = 24

resultsy= pd.DataFrame()



for j in range(600):

    dfr = initi[initi.sat_id==j][['y','type']]

    train = dfr[dfr.type=='train'].reset_index(drop=True)

    test = dfr[dfr.type=='test'].reset_index(drop=True)

    

    i=0

    

    while(i*24<len(test)):

        df = pd.concat((train, test), axis = 0, ignore_index=True)

        df['target'] = df.y

        features = []

        for period_mult in range(1,3,1): #два лага

            df["lag_period_{}".format(period_mult)] = df.target.shift(period_mult*lag_period)

            features.append("lag_period_{}".format(period_mult))

        df['lagf_mean'] = df[features].mean(axis = 1)

        features.extend(['lagf_mean'])

            

        model = LinearRegression()



        train_df = df[df.type=='train'][features + ['target']].dropna()

        test_df = df[df.type=='test'][features].reset_index(drop=True)

        test_df = test_df.loc[:23,:]

        

        model.fit(train_df.drop(['target'], axis = 1), train_df['target'])

        forecast = model.predict(test_df)

        test.loc[i*24:i*24+23,:]['y'] = forecast

        test.loc[i*24:i*24+23,:]['type'] = 'train'



        i+=1

    

    resultsy = pd.concat((resultsy, test),axis=0, ignore_index=True)
resultsy.drop(['type'], axis=1, inplace=True)
lag_period = 24

resultsz= pd.DataFrame()



for j in range(600):

    dfr = initi[initi.sat_id==j][['z','type']]

    train = dfr[dfr.type=='train'].reset_index(drop=True)

    test = dfr[dfr.type=='test'].reset_index(drop=True)

    

    i=0

    

    while(i*24<len(test)):

        df = pd.concat((train, test), axis = 0, ignore_index=True)

        df['target'] = df.z

        features = []

        for period_mult in range(1,3,1):

            df["lag_period_{}".format(period_mult)] = df.target.shift(period_mult*lag_period)

            features.append("lag_period_{}".format(period_mult))

        df['lagf_mean'] = df[features].mean(axis = 1)

        features.extend(['lagf_mean'])

            

        model = LinearRegression()



        train_df = df[df.type=='train'][features + ['target']].dropna()

    

        test_df = df[df.type=='test'][features].reset_index(drop=True)

        test_df = test_df.loc[:23,:]

        

        model.fit(train_df.drop(['target'], axis = 1), train_df['target'])

        forecast = model.predict(test_df)

        test.loc[i*24:i*24+23,:]['z'] = forecast

        test.loc[i*24:i*24+23,:]['type'] = 'train'



        i+=1

    

    resultsz = pd.concat((resultsz, test),axis=0, ignore_index=True)
resultsz.drop(['type'], axis=1, inplace=True)
all_results = pd.concat((resultsx,resultsy,resultsz), axis=1)
all_results.head()
df = pd.concat((all_results, initi[initi.type=='test'][['x_sim', 'y_sim', 'z_sim']].reset_index(drop=True) ), axis=1)
df.info()
df.loc[:243,:]
df['error'] = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)
submit = df[['error']]
submit.head()
len(submit)
sub = pd.read_csv('/kaggle/input/sputnik/sub.csv')
submit['id'] = (sub.reset_index()).id
submit
submit.index=sub.id
submit
submit.drop(['id'], axis=1, inplace=True)
submit
submit.to_csv('submit1.csv')
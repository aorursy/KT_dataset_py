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
df = pd.read_csv('/kaggle/input/bosch-production-line-performance/train_numeric.csv.zip',nrows=100)

df
pd.options.display.max_rows = 999

df.head()

from sklearn import linear_model



size = 100000

flag = 0

stations=[]

station_name='S0'

for i in pd.read_csv('/kaggle/input/bosch-production-line-performance/train_numeric.csv.zip',iterator=True, chunksize=size,compression='zip'):

    #reset all the values for next iteration

    print('new iteration')

    if (station_name == 'S52' ):

        #print('fun with flags')

        flag = 1

    a=i.columns

    station_counter = 0

    feature_counter = 0

    cumulative_feature = 0

    index=0

    station_name = 'S{}'.format(station_counter)

    while flag == 0 :

        feature_counter = 0

        while station_name in a[index+1]:

            index+=1

            feature_counter+=1

        print(feature_counter)

        cumulative_feature += feature_counter

        newframe = i.iloc[:,cumulative_feature-feature_counter+1:cumulative_feature+1]

        newframe.insert(0,"Id",i.iloc[ :,0],True)

        newframe = newframe.dropna( thresh = 2)

        newframe.to_csv(station_name+'_numeric.csv',index = False)

        station_counter+=1

        print(station_name)

        stations.append(station_name)

        station_name = 'S{}'.format(station_counter)

        if(station_name == 'S52'):

            print('passing to second is now')

            station_counter = 0

            feature_counter = 0

            cumulative_feature = 0

            break

        while station_name not in a[index+1]:

            print('wel wel wel')

            station_counter+=1

            station_name = 'S{}'.format(station_counter)

    while(flag == 1) and (len(i) != 0):

        feature_counter = 0

        while station_name in a[index+1]:

            index+=1

            feature_counter+=1

        cumulative_feature+=feature_counter

        newframe = i.iloc[:,cumulative_feature-feature_counter+1:cumulative_feature+1] 

        newframe.insert(0,"Id",i.iloc[ :,0],True)

        newframe = newframe.dropna(  thresh = 2)

        newframe.to_csv(station_name+'_numeric.csv',header = False,index = False,mode = 'a')

        station_counter+=1

        #print(station_name)

        station_name = 'S{}'.format(station_counter)

        if station_name =='S52':

            break

        while station_name not in a[index+1]:

            #print('wo')

            station_counter+=1

            station_name = 'S{}'.format(station_counter)

              

print('done done done')    
#empty consist of all the stations as keys and total number of missing data per station as values

empty = {}

for i in stations:

    df = pd.read_csv( i+'_numeric.csv')

    empty[i] = df.isna().sum().sum()

empty
terminal = stations[len(stations)-1]

# Feature Dictionnary 'FeatureName' : '# of Nan ' 

Features = {}

for i in stations:

    df = pd.read_csv( i+'_numeric.csv')

    counter = 1

    while(True):

        SerieOfFeature= df.iloc[:,counter]

        Null_per_Feature = SerieOfFeature.isnull().sum()

        Features[df.columns[counter]] = Null_per_Feature

        counter+=1

        if counter > len(df.columns)-1:

            break
Features
pd.read_csv('S1_numeric.csv').isna().sum().sum()
pd.read_csv('S5_numeric.csv')
df.isna().sum().sum()
df2=df.interpolate(method ='linear', limit_direction ='forward') 

df2.isna().sum().sum()
pd.read_csv('S1_numeric.csv')
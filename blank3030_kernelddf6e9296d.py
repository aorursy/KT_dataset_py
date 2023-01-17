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
import pandas as pd

import numpy as np



import matplotlib

import seaborn

import matplotlib.dates as md

from matplotlib import pyplot as plt



from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.cluster import KMeans

from sklearn.covariance import EllipticEnvelope

from sklearn.ensemble import IsolationForest

from sklearn.svm import OneClassSVM

from sklearn.metrics import f1_score

from sklearn.metrics import classification_report
df1 = pd.read_csv('/kaggle/input/training-dataset/ec2_cpu_utilization_77c1ca.csv')

df2 = pd.read_csv('/kaggle/input/training-dataset/ec2_cpu_utilization_53ea38.csv')

df3 = pd.read_csv('/kaggle/input/training-dataset/ec2_cpu_utilization_24ae8d.csv')

df4 = pd.read_csv('/kaggle/input/training-dataset/ec2_cpu_utilization_5f5533.csv')
ev1 = pd.read_csv('/kaggle/input/evaluation/ec2_cpu_utilization_77c1ca.csv')

ev2 = pd.read_csv('/kaggle/input/evaluation/elb_request_count_8c0756.csv')

ev3 = pd.read_csv('/kaggle/input/evaluation/ec2_disk_write_bytes_c0d644.csv')

ev4 = pd.read_csv('/kaggle/input/evaluation/ec2_cpu_utilization_53ea38.csv')

ev5 = pd.read_csv('/kaggle/input/evaluation/ec2_cpu_utilization_24ae8d.csv')

ev6 = pd.read_csv('/kaggle/input/evaluation/ec2_cpu_utilization_5f5533.csv')

ev7 = pd.read_csv('/kaggle/input/evaluation/ec2_disk_write_bytes_1ef3de.csv')

ev8 = pd.read_csv('/kaggle/input/evaluation/rds_cpu_utilization_cc0c53.csv')

ev9 = pd.read_csv('/kaggle/input/evaluation/ec2_network_in_257a54.csv')

ev10 = pd.read_csv('/kaggle/input/evaluation/ec2_network_in_5abac7.csv')

ev11 = pd.read_csv('/kaggle/input/evaluation/grok_asg_anomaly.csv')

ev12 = pd.read_csv('/kaggle/input/evaluation/rds_cpu_utilization_e47b3b.csv')
df3.head()
df1.head()
df4.info()
df1['timestamp'] = pd.to_datetime(df1['timestamp'])

df2['timestamp'] = pd.to_datetime(df2['timestamp'])

df3['timestamp'] = pd.to_datetime(df3['timestamp'])

df4['timestamp'] = pd.to_datetime(df4['timestamp'])


df1.plot(x='timestamp', y='value')

df2.plot(x='timestamp', y='value')

df3.plot(x='timestamp', y='value')

df4.plot(x='timestamp', y='value')
outliers_fraction = 0.01
def FeatEng(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # the hours and if it's night or day (7:00-22:00)

    df['hours'] = df['timestamp'].dt.hour

    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    # the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.

    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek

    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)

    # An estimation of anomly population of the dataset (necessary for several algorithm)

    outliers_fraction = 0.01

    # time with int to plot easily

    df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)

    return df
def CatFig(df):

    # creation of 4 distinct categories that seem useful (week end/day week & night/day)

    df['categories'] = df['WeekDay']*2 + df['daylight']



    a = df.loc[df['categories'] == 0, 'value']

    b = df.loc[df['categories'] == 1, 'value']

    c = df.loc[df['categories'] == 2, 'value']

    d = df.loc[df['categories'] == 3, 'value']



    fig, ax = plt.subplots()

    a_heights, a_bins = np.histogram(a)

    b_heights, b_bins = np.histogram(b, bins=a_bins)

    c_heights, c_bins = np.histogram(c, bins=a_bins)

    d_heights, d_bins = np.histogram(d, bins=a_bins)



    width = (a_bins[1] - a_bins[0])/6



    ax.bar(a_bins[:-1], a_heights*100/a.count(), width=width, facecolor='blue', label='WeekEndNight')

    ax.bar(b_bins[:-1]+width, (b_heights*100/b.count()), width=width, facecolor='green', label ='WeekEndLight')

    ax.bar(c_bins[:-1]+width*2, (c_heights*100/c.count()), width=width, facecolor='red', label ='WeekDayNight')

    ax.bar(d_bins[:-1]+width*3, (d_heights*100/d.count()), width=width, facecolor='black', label ='WeekDayLight')



    plt.legend()

    plt.show()
df3 = FeatEng(df3)

CatFig(df3)
def preprocess(df):

    data = df[['value', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]

    min_max_scaler = preprocessing.StandardScaler()

    np_scaled = min_max_scaler.fit_transform(data)

    data = pd.DataFrame(np_scaled)

    return data
alg1 =  IsolationForest(n_estimators = 1000, max_samples = 'auto', contamination = 0.1, bootstrap = True, behaviour = 'new')

alg2 = EllipticEnvelope(contamination = 0.096, support_fraction = 0.8)
def train_predict(df, alg):

    df = FeatEng(df)

    data = preprocess(df)

    alg.fit(data)

    df['anomaly'] = pd.Series(alg.predict(data))

    df['anomaly'] = df['anomaly'].map( {1: 0, -1: 1} )

    print(df['anomaly'].value_counts())

    # visualisation of anomaly throughout time 

    fig, ax = plt.subplots()

    c = df.loc[df['anomaly'] == 1, ['time_epoch', 'value']] #anomaly

    ax.plot(df['time_epoch'], df['value'], color='blue')

    ax.scatter(c['time_epoch'],c['value'], color='red')

    plt.show()

    print(f1_score(df['label'], df['anomaly']))

    print(classification_report(df['label'], df['anomaly']))

    return df
def test_predict(df, alg):

    df = FeatEng(df)

    data = preprocess(df)

    alg.fit(data)

    df['label'] = pd.Series(alg.predict(data))

    df['label'] = df['label'].map( {1: 0, -1: 1} )

    print(df['label'].value_counts()) 

    return df
df1 = train_predict(df1, alg2)

df2 = train_predict(df2, alg2)

df3 = train_predict(df3, alg2)

df4 = train_predict(df4, alg2)
ev4.head()
ev = [ev1, ev2, ev3, ev4, ev5, ev6, ev7, ev8, ev9, ev10, ev11, ev12]

for ev in ev:

    ev = test_predict(ev, alg2)
ev1 = ev1[['timestamp', 'value', 'label']]

ev2 = ev2[['timestamp', 'value', 'label']]

ev3 = ev3[['timestamp', 'value', 'label']]

ev4 = ev4[['timestamp', 'value', 'label']]

ev5 = ev5[['timestamp', 'value', 'label']]

ev6 = ev6[['timestamp', 'value', 'label']]

ev7 = ev7[['timestamp', 'value', 'label']]

ev8 = ev8[['timestamp', 'value', 'label']]

ev9 = ev9[['timestamp', 'value', 'label']]

ev10 = ev10[['timestamp', 'value', 'label']]

ev11 = ev11[['timestamp', 'value', 'label']]

ev12 = ev12[['timestamp', 'value', 'label']]
ev1.to_csv('ev1.csv', index = False)

ev2.to_csv('ev2.csv', index = False)

ev3.to_csv('ev3.csv', index = False)

ev4.to_csv('ev4.csv', index = False)

ev5.to_csv('ev5.csv', index = False)

ev6.to_csv('ev6.csv', index = False)

ev7.to_csv('ev7.csv', index = False)

ev8.to_csv('ev8.csv', index = False)

ev9.to_csv('ev9.csv', index = False)

ev10.to_csv('ev10.csv', index = False)

ev11.to_csv('ev11.csv', index = False)

ev12.to_csv('ev12.csv', index = False)
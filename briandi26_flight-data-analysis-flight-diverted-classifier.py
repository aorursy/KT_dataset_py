# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime
# loading dataset



df = pd.read_csv('../input/2008.csv')

df.head()
df.shape
# mengecek nilai yang hilang (NaN)



data_nan = df.isnull().sum(axis=0).reset_index()

data_nan.columns = ['variable', 'missing values']

data_nan
data_nan['filling factor (%)'] = (df.shape[0] - data_nan['missing values'])/df.shape[0]*100.

data_nan
data_nan.sort_values(by='filling factor (%)').reset_index(drop=True)
df['Date'] = pd.to_datetime(df['Year'].map(str)+'-'+df['Month'].map(str)+'-'+df['DayofMonth'].map(str))

df.head()
def format_tanggal (dataframe):

    if pd.isnull(dataframe):

        return np.nan

    else:

        if dataframe == 2400: dataframe = 0

        dataframe = "{0:04d}".format(int(dataframe))

        tanggal = datetime.time(int(dataframe[0:2]), int(dataframe[2:4]))

        return tanggal



def combine_date(x):

    if pd.isnull(x[0]) or pd.isnull(x[1]):

        return np.nan

    else:

        return datetime.datetime.combine(x[0],x[1])



def create_flight_time(data, col):    

    liste = []

    for index, cols in data[['Date', col]].iterrows():    

        if pd.isnull(cols[1]):

            liste.append(np.nan)

        elif float(cols[1]) == 2400:

            cols[0] += datetime.timedelta(days=1)

            cols[1] = datetime.time(0,0)

            liste.append(combine_date(cols))

        else:

            cols[1] = format_tanggal(cols[1])

            liste.append(combine_date(cols))

    return pd.Series(liste)
df['CRSDepTime'] = create_flight_time(df, 'CRSDepTime')

df['DepTime'] = df['DepTime'].apply(format_tanggal)

df['CRSArrTime'] = df['CRSArrTime'].apply(format_tanggal)

df['ArrTime'] = df['ArrTime'].apply(format_tanggal)
df.loc[:5, ['CRSDepTime', 'CRSArrTime', 'DepTime',

             'ArrTime', 'DepDelay', 'ArrDelay']]
# mengekstrak parameter statistik dari fungsi groupby

def get_stats(group):

    return {'min': group.min(), 'max': group.max(),

            'count': group.count(), 'mean': group.mean()}



# membuat dataframe dengan info statistik dari setiap pesawat

global_stats = df['DepDelay'].groupby(df['UniqueCarrier']).apply(get_stats).unstack()

global_stats = global_stats.sort_values('count')

global_stats
# mengelompokkan penerbangan yang mengalami delay

delay_type = lambda x:((0,1)[x > 5],2)[x > 45]

df['DelayLvl'] = df['DepDelay'].apply(delay_type)



fig = plt.figure(1, figsize=(10,7))

ax = sns.countplot(y="UniqueCarrier", hue='DelayLvl', data=df)



# mengatur label dari plot yang akan dibuat

plt.setp(ax.get_xticklabels(), fontsize=12, weight = 'normal', rotation = 0);

plt.setp(ax.get_yticklabels(), fontsize=12, weight = 'bold', rotation = 0);

ax.yaxis.label.set_visible(False)

plt.xlabel('Flight count', fontsize=16, weight = 'bold', labelpad=10)



# mengatur legenda dari plot yang akan dibuat

L = plt.legend()

L.get_texts()[0].set_text('on time (t < 5 min)')

L.get_texts()[1].set_text('small delay (5 < t < 45 min)')

L.get_texts()[2].set_text('large delay (t > 45 min)')

plt.show()
df.columns
np.where(df.dtypes.values == np.dtype('float64'))
new_df = df.iloc[:, np.where(df.dtypes.values == np.dtype('float64'))[0]]

new_df.head()
for i in range(len(new_df.columns)):

    if (new_df.isnull().iloc[:,i].shape[0]>0):

        print('\nAttribute-',i,' (before) :',new_df.isnull().iloc[:,i].shape[0])

        new_df.iloc[:,i].fillna(new_df.iloc[:,i].mean(), inplace=True)

        print('\nAttribute-',i,' (after) :',new_df.isnull().iloc[:,i].shape[0])
new_df.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score
# define the dictionary of models our script can use

# the key to the dictionary is the name of the model

# (supplied via command line argument) and the value is the model itself

models = {

    "knn": KNeighborsClassifier(n_neighbors=1),

    "naive_bayes": GaussianNB(),

    "logit": LogisticRegression(solver="lbfgs", multi_class="auto"),

    "svm": SVC(kernel="linear", gamma="auto"),

    "decision_tree": DecisionTreeClassifier(),

    "random_forest": RandomForestClassifier(n_estimators=100),

    'mlp': MLPClassifier()

}
df.Diverted
# perform a training testing split, using 75% of the data for

# training and 25% for evaluation

(trainX, testX, trainY, testY) = train_test_split(new_df.values, df.Diverted, random_state=3, test_size=0.25)
# train the Random Forest model

print("[INFO] using '{}' model".format("random_forest"))

model = models["random_forest"]

model.fit(trainX, trainY)

# make predictions on our data and show a classification report

print("[INFO] evaluating...")

predictions = model.predict(testX)

print(classification_report(testY, predictions))
accuracy_score(testY, predictions)
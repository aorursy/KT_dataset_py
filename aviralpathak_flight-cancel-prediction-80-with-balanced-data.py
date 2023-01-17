# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#to display all the columns

pd.set_option('display.max_columns', None)
#loading datasets

df1 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')

df2 = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')
#concatenating both the datasets

frames = [df1, df2]



df = pd.concat(frames)



df.reset_index(drop=True, inplace=True)
#dealing with empty rows

df.drop(['Unnamed: 21'], axis=1, inplace=True)

df = df.fillna(method ='pad')
#differentiating distance 



low_dist = df[(df['DISTANCE']<=1000) & (df['CANCELLED']==1.0)]['CANCELLED'].count()

mid_dist = df[((df['DISTANCE']> 1000) & (df['DISTANCE']<=2000)) & (df['CANCELLED']==1.0)]['CANCELLED'].count()

high_dist = df[(df['DISTANCE']> 2000) & (df['CANCELLED']==1.0)]['CANCELLED'].count()
df[(df['DISTANCE']<=1000)& (df['CANCELLED']==0)]
#plotting distance into pie-chart



labels = ['low','mid','high']

data  = [low_dist,mid_dist,high_dist]



plt.rcParams['figure.figsize'] = (15,9)



plt.pie(data, labels = labels,explode=(0, 0.3, 0.9),autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title("Canceled Hotel Percent by Type", fontsize=20)

plt.show()
#differentiating departure time

low_dep = df[(df['DEP_TIME']<=1000) & (df['CANCELLED']==1.0)]['CANCELLED'].count()

mid_dep = df[((df['DEP_TIME']> 1000) & (df['DEP_TIME']<=2000)) & (df['CANCELLED']==1.0)]['CANCELLED'].count()

high_dep = df[(df['DEP_TIME']> 2000) & (df['CANCELLED']==1.0)]['CANCELLED'].count()
labels = ['low','mid','high']

data  = [low_dep,mid_dep,high_dep]



plt.rcParams['figure.figsize'] = (15,9)



plt.pie(data, labels = labels,explode=(0, 0.1, 0.1),autopct='%1.1f%%',shadow=True)

plt.axis('equal')

plt.title("Canceled Hotel Percent by Type", fontsize=20)

plt.show()
#data with cancelled flights



cancel =  df[df['CANCELLED']==1.0]
cancel.groupby('DEP_TIME').count()
#day with most number of cancelled flights



month_cancel = cancel.groupby('DAY_OF_MONTH')['CANCELLED'].count()

sns.barplot(x = month_cancel.index , y = month_cancel.values,palette = 'bone')
def cancel_func(column):

    org_cancel = cancel.groupby(column)['CANCELLED'].count()

    org_cancel = org_cancel[org_cancel >400]

    sns.barplot(x = org_cancel.index , y = org_cancel.values)
#flight cancelation by origin

cancel_func('ORIGIN')
#flight cancelation by origin

cancel_func('DEST')
#cancelation by op_carrier

carrier_cancel = cancel.groupby('OP_CARRIER')['CANCELLED'].count()

sns.barplot(x = carrier_cancel.index , y = carrier_cancel.values, palette = 'Wistia')
df1 = df[['OP_CARRIER','ORIGIN','DEST','DAY_OF_MONTH','DEP_TIME','ARR_TIME','DISTANCE','CANCELLED']]

df1.head()
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()

df1['OP_CARRIER']= label_encoder.fit_transform(df1['OP_CARRIER'])

df1['ORIGIN']= label_encoder.fit_transform(df1['ORIGIN'])

df1['DEST']= label_encoder.fit_transform(df1['DEST'])
X = df1.drop('CANCELLED', axis=1)

y = df1.CANCELLED
from imblearn.under_sampling import NearMiss
nm = NearMiss()
X_res, y_res = nm.fit_sample(X,y.ravel())
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(

    X_res, y_res, test_size=0.3, random_state=101)
from sklearn.tree import DecisionTreeClassifier

algo = DecisionTreeClassifier()

algo.fit(X_train, y_train)
predict_test = algo.predict(X_test)
accuracy_score(y_test,predict_test)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 8)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy_score(y_test,y_pred)
from sklearn.naive_bayes import GaussianNB

GNBclf = GaussianNB()

model = GNBclf.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test,y_pred)
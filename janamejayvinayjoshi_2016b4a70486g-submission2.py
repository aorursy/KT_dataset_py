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
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import sklearn

from scipy import stats

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set() # for plot styling

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin

from sklearn.cluster import MiniBatchKMeans
import time

import warnings



import numpy as np

import matplotlib.pyplot as plt



from sklearn import cluster, datasets

from sklearn.preprocessing import StandardScaler

from itertools import cycle, islice
df_train_path = '/kaggle/input/eval-lab-3-f464/train.csv'

df_test_path = '/kaggle/input/eval-lab-3-f464/test.csv'

df_tr = pd.read_csv(df_train_path)

df_te = pd.read_csv(df_test_path)

print(df_tr.info())

print(df_te.info())
#Number of unique values in different columns

print("Gender =",df_tr['gender'].nunique())

print("SeniorCitizen =",df_tr['SeniorCitizen'].nunique())

print("Married =",df_tr['Married'].nunique())

print("Children =",df_tr['Children'].nunique())

print("TVConnection =",df_tr['TVConnection'].nunique())

print("Channel1 =",df_tr['Channel1'].nunique())

print("Channel2 =",df_tr['Channel2'].nunique())

print("Channel3 =",df_tr['Channel3'].nunique())

print("Channel4 =",df_tr['Channel4'].nunique())

print("Channel5 =",df_tr['Channel5'].nunique())

print("Channel6 =",df_tr['Channel6'].nunique())

print("Internet =",df_tr['Internet'].nunique())

print("HighSpeed =",df_tr['HighSpeed'].nunique())

print("AddedServices =",df_tr['AddedServices'].nunique())

print("Subscription =",df_tr['Subscription'].nunique())

print("PaymentMethod =",df_tr['PaymentMethod'].nunique())
#Value_Counts of different columns

print(df_tr['gender'].value_counts())

print(df_tr['SeniorCitizen'].value_counts())

print(df_tr['Married'].value_counts())

print(df_tr['Children'].value_counts())

print(df_tr['TVConnection'].value_counts())

print(df_tr['Channel1'].value_counts())

print(df_tr['Channel2'].value_counts())

print(df_tr['Channel3'].value_counts())

print(df_tr['Channel4'].value_counts())

print(df_tr['Channel5'].value_counts())

print(df_tr['Channel6'].value_counts())

print(df_tr['Internet'].value_counts())

print(df_tr['HighSpeed'].value_counts())

print(df_tr['AddedServices'].value_counts())

print(df_tr['Subscription'].value_counts())

print(df_tr['PaymentMethod'].value_counts())
def onezeroconv(row):

    #GENDER

    if row.gender == 'Female':

        row.gender = 0

    elif row.gender == 'Male':

        row.gender = 1

        

    #MARRIED

    if row.Married == 'No':

        row.Married = 0

    elif row.Married == 'Yes':

        row.Married = 1

        

    #CHILDREN

    if row.Children == 'No':

        row.Children = 0

    elif row.Children == 'Yes':

        row.Children = 1

        

    #TVCONNECTION

    if row.TVConnection == 'No':

        row.TVConnection = 0

    elif row.TVConnection == 'Cable':

        row.TVConnection = 1

    elif row.TVConnection == 'DTH':

        row.TVConnection = 2

    

    #CHANNEL1

    if row.Channel1 == 'No':

        row.Channel1 = 0

    elif row.Channel1 == 'Yes':

        row.Channel1 = 1

    elif row.Channel1 == 'No tv connection':

        row.Channel1 = 2

        

    #CHANNEL2

    if row.Channel2 == 'No':

        row.Channel2 = 0

    elif row.Channel2 == 'Yes':

        row.Channel2 = 1

    elif row.Channel2 == 'No tv connection':

        row.Channel2 = 2

        

    #CHANNEL3

    if row.Channel3 == 'No':

        row.Channel3 = 0

    elif row.Channel3 == 'Yes':

        row.Channel3 = 1

    elif row.Channel3 == 'No tv connection':

        row.Channel3 = 2

        

    #CHANNEL4

    if row.Channel4 == 'No':

        row.Channel4 = 0

    elif row.Channel4 == 'Yes':

        row.Channel4 = 1

    elif row.Channel4 == 'No tv connection':

        row.Channel4 = 2

        

    #CHANNEL5

    if row.Channel5 == 'No':

        row.Channel5 = 0

    elif row.Channel5 == 'Yes':

        row.Channel5 = 1

    elif row.Channel5 == 'No tv connection':

        row.Channel5 = 2

        

    #CHANNEL6

    if row.Channel6 == 'No':

        row.Channel6 = 0

    elif row.Channel6 == 'Yes':

        row.Channel6 = 1

    elif row.Channel6 == 'No tv connection':

        row.Channel6 = 2

        

    #INTERNET

    if row.Internet == 'No':

        row.Internet = 0

    elif row.Internet == 'Yes':

        row.Internet = 1

    

    #HIGHSPEED

    if row.HighSpeed == 'No':

        row.HighSpeed = 0

    elif row.HighSpeed == 'Yes':

        row.HighSpeed = 1

    elif row.HighSpeed == 'No internet':

        row.HighSpeed = 2

        

    #ADDEDSERVICES

    if row.AddedServices == 'No':

        row.AddedServices = 0

    elif row.AddedServices == 'Yes':

        row.AddedServices = 1

        

    #SUBSCRIPTION

    if row.Subscription == 'Monthly':

        row.Subscription = 0

    elif row.Subscription == 'Biannually':

        row.Subscription = 1

    elif row.Subscription == 'Annually':

        row.Subscription = 2

    

    #PAYMENTMETHOD

    if row.PaymentMethod == 'Net Banking':

        row.PaymentMethod = 0

    elif row.PaymentMethod == 'Cash':

        row.PaymentMethod = 1

    elif row.PaymentMethod == 'Bank transfer':

        row.PaymentMethod = 2

    elif row.PaymentMethod == 'Credit card':

        row.PaymentMethod = 3

    return row
df_tr = df_tr.apply((lambda x: onezeroconv(x)), axis = 1)

df_te = df_te.apply((lambda x: onezeroconv(x)), axis = 1)
def onehot(row):

    #GENDER

    if row.gender == 'Female':

        row.gender = 0

    elif row.gender == 'Male':

        row.gender = 1

        

    #MARRIED

    if row.Married == 'No':

        row.Married = 0

    elif row.Married == 'Yes':

        row.Married = 1

        

    #CHILDREN

    if row.Children == 'No':

        row.Children = 0

    elif row.Children == 'Yes':

        row.Children = 1

        

    #TVCONNECTION 'Cable', 'DTH'

    if row.TVConnection == 1:

        row['Cable'] = 1

        row['DTH'] = 0

    elif row.TVConnection == 2:

        row['Cable'] = 0

        row['DTH'] = 1

        row.TVConnection = 1

    elif row.TVConnection == 0:

        row['Cable'] = 0

        row['DTH'] = 0

    

    #CHANNEL1

    if row.Channel1 == 2:

        row.Channel1 = 1

        

    #CHANNEL2

    if row.Channel2 == 2:

        row.Channel2 = 1

        

    #CHANNEL3

    if row.Channel3 == 2:

        row.Channel3 = 1

        

    #CHANNEL4

    if row.Channel4 == 2:

        row.Channel4 = 1

        

    #CHANNEL5

    if row.Channel5 == 2:

        row.Channel5 = 1

        

    #CHANNEL6

    if row.Channel6 == 2:

        row.Channel6 = 1

        

    #INTERNET

    if row.Internet == 'No':

        row.Internet = 0

    elif row.Internet == 'Yes':

        row.Internet = 1

    

    #HIGHSPEED

    if row.HighSpeed == 2:

        row.HighSpeed = 1

        

    #ADDEDSERVICES

    if row.AddedServices == 'No':

        row.AddedServices = 0

    elif row.AddedServices == 'Yes':

        row.AddedServices = 1

        

    #SUBSCRIPTION 'Monthly','Biannually','Annually'

    if row.Subscription == 0:

        row['Monthly'] = 1

        row['Binannually'] = 0

        row['Annually'] = 0

    elif row.Subscription == 1:

        row['Monthly'] = 0

        row['Binannually'] = 1

        row['Annually'] = 0

    elif row.Subscription == 2:

        row['Monthly'] = 0

        row['Binannually'] = 0

        row['Annually'] = 1

    

    #PAYMENTMETHOD 'Net Banking', 'Cash', 'Bank transfer', 'Credit card'

    if row.PaymentMethod == 0:

        row['Net Banking'] = 1

        row['Cash'] = 0

        row['Bank transfer'] = 0

        row['Credit card'] = 0

    elif row.PaymentMethod == 1:

        row['Net Banking'] = 0

        row['Cash'] = 1

        row['Bank transfer'] = 0

        row['Credit card'] = 0

    elif row.PaymentMethod == 2:

        row['Net Banking'] = 0

        row['Cash'] = 0

        row['Bank transfer'] = 1

        row['Credit card'] = 0

    elif row.PaymentMethod == 3:

        row['Net Banking'] = 0

        row['Cash'] = 0

        row['Bank transfer'] = 1

        row['Credit card'] = 1

    

    return row 
df_tr = df_tr.apply((lambda x: onehot(x)), axis = 1)

df_te = df_te.apply((lambda x: onehot(x)), axis = 1)
x_tr = 0

count_tr = 0

for index,row in df_tr.iterrows():

    if row['TotalCharges'] != " ":

        x_tr += float(row['TotalCharges'])

        count_tr += 1

mtc_tr = float(x_tr/count_tr)



x_te = 0

count_te = 0

for index,row in df_te.iterrows():

    if row['TotalCharges'] != " ":

        x_te += float(row['TotalCharges'])

        count_te += 1

mtc_te = float(x_te/count_te)
def repmean(row,mtc):

    if row.TotalCharges == " ":

        row.TotalCharges = mtc

    return row



df_tr = df_tr.apply((lambda x: repmean(x,mtc_tr)), axis = 1)

df_te = df_te.apply((lambda x: repmean(x,mtc_te)), axis = 1)
sns.boxplot(x="Satisfied", y="MonthlyCharges", data=df_tr)
sns.boxplot(x="Satisfied", y="tenure", data=df_tr)
df_tr['TotalCharges'] = pd.to_numeric(df_tr['TotalCharges'])

sns.boxplot(x="Satisfied", y="MonthlyCharges", data=df_tr)
cat_feat = ['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Annually','Binannually','Monthly','Net Banking','Cash','Credit card','Bank transfer','Cable','DTH','Satisfied']

cat_feat_corr = df_tr[cat_feat].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).copy()
cat_feat_corr['Satisfied'].sort_values()
#So here itself we declare 3 feature sets : cat_feat1 (All), cat_feat2 (>0.1 and <-0.1), cat_feat3 (>0.15 and <-0.1)

cat_feat1 = cat_feat.copy()

cat_feat2 = ['Net Banking','SeniorCitizen','Channel1','Channel2','Credit card','Married','Children','Annually','AddedServices','Bank transfer','TVConnection','Channel3','Channel4','Channel5','Channel6','Binannually','Monthly']

cat_feat3 = ['Net Banking','Married','Children','Annually','AddedServices','Bank transfer','TVConnection','Channel3','Channel4','Channel5','Channel6','Binannually','Monthly']
cat_feat1.append('tenure')

cat_feat1.append('MonthlyCharges')

cat_feat1.append('TotalCharges')



cat_feat2.append('tenure')

cat_feat2.append('MonthlyCharges')

cat_feat2.append('TotalCharges')



cat_feat3.append('tenure')

cat_feat3.append('MonthlyCharges')

cat_feat3.append('TotalCharges')
x1 = df_tr[cat_feat1].copy()

x2 = df_tr[cat_feat2].copy()

x3 = df_tr[cat_feat3].copy()



cat_feat1_te = ['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Annually','Binannually','Monthly','Net Banking','Cash','Credit card','Bank transfer','Cable','DTH','MonthlyCharges','TotalCharges','tenure']



y1 = df_te[cat_feat1_te].copy()

y2 = df_te[cat_feat2].copy()

y3 = df_te[cat_feat3].copy()
from sklearn.preprocessing import RobustScaler

scaler1 = RobustScaler()

scaler2 = RobustScaler()

scaler3 = RobustScaler()

num_feat = ['MonthlyCharges','TotalCharges','tenure']



x1[num_feat] = scaler1.fit_transform(x1[num_feat])

y1[num_feat] = scaler1.transform(y1[num_feat])



x2[num_feat] = scaler2.fit_transform(x2[num_feat])

y2[num_feat] = scaler2.transform(y2[num_feat])



x3[num_feat] = scaler3.fit_transform(x3[num_feat])

y3[num_feat] = scaler3.transform(y3[num_feat])
### K-MEANS, cat_feat1

kmeans1 = KMeans(n_clusters=2)

kmeans1.fit_predict(x1[cat_feat1_te])

x1_kmeans1 = kmeans1.predict(x1[cat_feat1_te])

y1_kmeans1 = kmeans1.predict(y1)
cust_id_km1 = df_te['custId'].copy()

output_km1 = pd.DataFrame(columns=['custId','Satisfied'])

output_km1['custId'] = cust_id_km1

output_km1['Satisfied'] = y1_kmeans1
output_km1.to_csv("km1.csv",index=False)
### K-MEANS, cat_feat2

kmeans3 = KMeans(n_clusters=2)

kmeans3.fit_predict(x3)

x3_kmeans3 = kmeans3.predict(x3)

y3_kmeans3 = kmeans3.predict(y3)
cust_id_km3 = df_te['custId'].copy()

output_km3 = pd.DataFrame(columns=['custId','Satisfied'])

output_km3['custId'] = cust_id_km3

output_km3['Satisfied'] = y3_kmeans3
output_km3.to_csv("km3.csv",index=False)
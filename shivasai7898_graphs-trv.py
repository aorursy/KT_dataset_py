# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import math

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error

from sklearn import preprocessing

import matplotlib.gridspec as gridspec

from numpy import median

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# load data to pandas dataframe

district_wise_rainfall = pd.read_csv('../input/scmp2k19.csv')

district_wise_rainfall.head()
# Let's find number of states we have

subdivision = district_wise_rainfall['district'].unique()

subdivision
plt.style.use('ggplot')



fig = plt.figure(figsize=(18, 28))

ax = plt.subplot(2,1,1)

ax = plt.xticks(rotation=90)

ax = sns.boxplot(x='district', y='humidity_max', data=district_wise_rainfall)

ax = plt.title('Annual rainfall in all Districts')



ax = plt.subplot(2,1,2)

ax = plt.xticks(rotation=90)

ax = sns.barplot(x='district', y='humidity_max', data=district_wise_rainfall)
total_rainfall_in_districts = district_wise_rainfall.groupby(['district']).sum()

total_rainfall_in_districts['district'] = total_rainfall_in_districts.index

total_rainfall_in_districts
plt.style.use('ggplot')

index = total_rainfall_in_districts.index

fig = plt.figure(figsize=(18, 28))

ax1 = plt.subplot2grid((2, 2), (0, 0))

ax1 = plt.xticks(rotation=90)

ax1 = sns.heatmap(total_rainfall_in_districts[['Aws ID','cumm_rainfall','humidity_min','humidity_max']])

ax1 = plt.title('Total Rainfall Annual')



ax2 = plt.subplot2grid((2, 2), (0, 1))

ax2 = plt.xticks(rotation=90)

ax2 = sns.heatmap(total_rainfall_in_districts[['Aws ID','cumm_rainfall','humidity_min','humidity_max']])

ax2 = plt.title('Total Rainfall Seasonal')



ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax3 = plt.xticks(rotation=90)

ax3 = sns.barplot(x='district', y='humidity_max', data=total_rainfall_in_districts.sort_values('humidity_max'))

ax3 = plt.title('Total Rainfall in all districts in increasing order')
fig = plt.figure(figsize=(18, 28))

ax1 = plt.subplot2grid((2, 2), (0, 0))

ax1 = plt.xticks(rotation=90)

ax1 = sns.heatmap(total_rainfall_in_districts[['Aws ID','cumm_rainfall','humidity_min','humidity_max']])

ax1 = plt.title('Total Rainfall Annual in all years')



ax2 = plt.subplot2grid((2, 2), (0, 1))

ax2 = plt.xticks(rotation=90)

ax2 = sns.heatmap(total_rainfall_in_districts[['Aws ID','cumm_rainfall','humidity_min','humidity_max']])

ax2 = plt.title('Total Rainfall Seasonal in all years')



ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

ax3 = plt.xticks(rotation=90)

ax3 = sns.barplot(x='Aws ID', y='humidity_max', data=total_rainfall_in_districts)

ax3 = plt.title('Total Rainfall in  all years')
x1 = total_rainfall_in_districts.sort_values('humidity_max')

fig = plt.figure(figsize=(20, 8))

ax = plt.xticks(rotation=90)

ax = x1['humidity_max'].plot.bar(color=['red', 'pink', 'darkred'], edgecolor = 'black')

ax = plt.title('Highest and Lowest rainfalls')
x2 = total_rainfall_in_districts.sort_values('humidity_min')

fig = plt.figure(figsize=(20, 8))

ax = plt.xticks(rotation=90)

ax = x2['humidity_min'].plot.bar(color=['red', 'pink', 'darkred'], edgecolor = 'black')

ax = plt.title('Highest and Lowest rainfalls')
total_rainfall_in_districts.info()
annualy_total_rainfall = total_rainfall_in_districts.head(12)

seasonal_total_rainfall = total_rainfall_in_districts.tail(4)

# For each state, we visualize the rainfall patterns in different months and season

plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(40,100))

for i in range(30):

    plt.subplot(6, 5, i+1)

    t = annualy_total_rainfall[annualy_total_rainfall.columns[i]].plot.bar()

    t.set_title("Annualy Rainfall for " + str(annualy_total_rainfall.columns[i]))

plt.show()
# For each state, we visualize the rainfall patterns in different months and season

plt.style.use('seaborn-whitegrid')

fig = plt.figure(figsize=(40,100))



for i in range(31):

    plt.subplot(6, 5, i+1)

    t = seasonal_total_rainfall[seasonal_total_rainfall.columns[i]].plot.bar()

    t.set_title("Seasonal Rainfall for " +str(seasonal_total_rainfall.columns[i]))

plt.show()
district_wise_rainfall.hist(figsize =(30,30))
# distribution of rainfall over the district

district_wise_rainfall.groupby("district").sum()['humidity_max'].plot(figsize=(10,5))

plt.xlabel('district',fontsize=15)

plt.ylabel('Seasonal Rainfall (in mm)',fontsize=15)

plt.title('Seasonal Rainfall from Year 2018',fontsize=15)

plt.grid()

plt.ioff()
district_wise_rainfall[['Aws ID','cumm_rainfall','humidity_min','humidity_max']].groupby("Aws ID").mean().plot.barh(stacked=True,figsize=(15,7))

plt.ylabel('Aws ID',fontsize=15)

plt.xlabel('Seasonal Rainfall (in mm)',fontsize=15)

plt.title('Seasonal Rainfall from Year 2018',fontsize=15)

plt.grid()

plt.ioff()
district_wise_rainfall[['district','Aws ID','cumm_rainfall','humidity_min','humidity_max']].groupby("district").mean().plot.barh(stacked=True,figsize=(15,7))

plt.ylabel('district',fontsize=15)

plt.xlabel('Seasonal Rainfall (in mm)',fontsize=15)

plt.title('Seasonal Rainfall from Year 2018',fontsize=15)

plt.grid()

plt.ioff()
plt.figure(figsize=(15,6))

sns.heatmap(district_wise_rainfall[['district','Aws ID','cumm_rainfall','humidity_min','humidity_max']].corr(),annot=True)

plt.show()
df2=district_wise_rainfall
df2.head()
df3 = df2.sort_values('district')

annual_array = []

for element in range(0,4116,116):

    annual_array.append(df2.loc[element,"humidity_min"])

    print(element,df2.loc[element,"district"])
#Changing district to unique numbers 

df2.district.unique()

dictionary = {}

for c, value in enumerate(df2.district.unique(), 1):

    #print(c, value)

    dictionary[value] = c

print(dictionary)

df2["district"] = df2.district.map(dictionary)

df2.head()
df2.columns
df2.dropna(inplace=True)

df2.isnull().sum().sum()
df2.replace([np.inf, -np.inf], np.nan, inplace = True)

df3 = df2

df3.head()
#Select features for taining

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing 

from sklearn.pipeline import Pipeline

df2.head(0)

features = []

for element in df2.head(0):

    features.append(element)

features.remove('district')

features.remove('mandal')

features.remove('location')

features.remove('cumm_rainfall')

features.remove('humidity_max')

features.remove('Aws ID')

print(features)
#Target feature (to predict humidity_max rainfall

features2 = ['humidity_max']



# Separating the features

x  = df2.loc[:, features].values

# Separating the target

y = df2.loc[:,features2].values

# Standardizing the features
#splitting training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#SVM

from sklearn import svm

from sklearn.datasets import make_classification

clf = svm.SVC()

X_train, y_train = make_classification()

X_test, y_test = make_classification()

clf.fit(X_train, y_train)
print(clf.intercept_)
clf.predict(X_test)
clf.score(X_test, y_test, sample_weight=None)
y_score = clf.decision_function(X_test)
# Random forest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train, y_train)

clf.predict(X_test)

clf.score(X_test, y_test)
# prediction
df4=df3
df4.groupby('district').size()
print("Co-Variance =",df4.cov())

print("Co-Relation =",df4.corr())
corr_cols=df4.corr()['humidity_min'].sort_values()[::-1]

print("Index of correlation columns:",corr_cols.index)
#Linear Model Fitted to Each Subdivision Category Independently 
import sklearn.linear_model as sk

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import metrics



df2 = df4[['Aws ID','cumm_rainfall','humidity_min','humidity_max']]

df2.columns = np.array(['Aws ID', 'x1','x2','x3'])



for k in range(1,9):

    df3 = df4[['Aws ID','cumm_rainfall','humidity_min','humidity_max']]

    df3.columns = np.array(['Aws ID', 'x1','x2','x3'])

    df2 = df2.append(df3)

df2.index = range(df2.shape[0])

    

#df2 = pd.concat([df2, pd.get_dummies(df2['SUBDIVISION'])], axis=1)



df2.drop('Aws ID', axis=1,inplace=True)

#print(df2.info())

msk = np.random.rand(len(df2)) < 0.8



df_train = df2[msk]

df_test = df2[~msk]

df_train.index = range(df_train.shape[0])

df_test.index = range(df_test.shape[0])



reg =sk.LinearRegression()

reg.fit(df_train.drop('x1',axis=1),df_train['x1'])

predicted_values = reg.predict(df_train.drop('x1',axis=1))

residuals = predicted_values-df_train['x1'].values

print('MAD (Training Data): ' + str(np.mean(np.abs(residuals))))

df_res = pd.DataFrame(residuals)

df_res.columns = ['Residuals']



fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res.plot.line(title='Different b/w Actual and Predicted (Training Data)', color = 'c', ax=ax,fontsize=20)

ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)





predicted_values = reg.predict(df_test.drop('x1',axis=1))

residuals = predicted_values-df_test['x1'].values

print('MAD (Test Data): ' + str(np.mean(np.abs(residuals))))

df_res = pd.DataFrame(residuals)

df_res.columns = ['Residuals']



fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res.plot.line(title='Different b/w Actual and Predicted (Test Data)', color='m', ax=ax,fontsize=20)

ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)
df_res_training = pd.DataFrame(columns=np.array(['Residuals']))

df_res_testing = pd.DataFrame(columns=np.array(['Residuals']))

list_mad_training = []

mean_abs_diff_training = 0

list_mad_testing = []

mean_abs_diff_testing = 0

for subd in subdivs:

    df1 = df[df['Aws ID']==awsis]    

    df2 = df1[['cumm_rainfall','humidity_min','humidity_max']]

    df2.columns = np.array(['x1','x2','x3'])

    for k in range(1,9):

        df3 = df1[['cumm_rainfall','humidity_min','humidity_max']]

        df3.columns = np.array(['x1','x2','x3'])

        df2 = df2.append(df6)

    df2.index = range(df2.shape[0])

    msk = np.random.rand(len(df2)) < 0.8

    df_train = df2[msk]

    df_test = df2[~msk]

    df_train.index = range(df_train.shape[0])

    df_test.index = range(df_test.shape[0])

    reg = linear_model.LinearRegression()

    reg.fit(df_train.drop('x1',axis=1),df_train['x1'])

    

    predicted_values = reg.predict(df_train.drop('x1',axis=1))

    residuals = predicted_values-df_train['x1'].values

    df_res_training = df_res_training.append(pd.DataFrame(residuals,columns=np.array(['Residuals'])))

    mean_abs_diff_training = mean_abs_diff_training + np.sum(np.abs(residuals))

    list_mad_training.append(np.mean(np.abs(residuals)))

    

    predicted_values = reg.predict(df_test.drop('x1',axis=1))

    residuals = predicted_values-df_test['x1'].values

    df_res_testing = df_res_testing.append(pd.DataFrame(residuals,columns=np.array(['Residuals'])))

    mean_abs_diff_testing = mean_abs_diff_testing + np.sum(np.abs(residuals))

    list_mad_testing.append(np.mean(np.abs(residuals)))

    

    

df_res_training.index = range(df_res_training.shape[0])

mean_abs_diff_training = mean_abs_diff_training/df_res_training.shape[0]

print('Overall MAD (Training): ' + str(mean_abs_diff_training))

fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res_training.plot.line(title='Residuals (Training)', color='c',ax=ax,fontsize=20)

#ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)



df_res_testing.index = range(df_res_testing.shape[0])

mean_abs_diff_testing = mean_abs_diff_testing/df_res_testing.shape[0]

print('Overall MAD (Testing): ' + str(mean_abs_diff_testing))

fig = plt.figure(figsize=(18,10))

ax = fig.add_subplot(111)

df_res_testing.plot.line(title='Residuals (Testing)', color='m',ax=ax,fontsize=20)

#ax.xaxis.set_ticklabels([])

plt.ylabel('Residual')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)





pd_mad = pd.DataFrame(data=list_mad_training,columns=["MAD (Train)"])

pd_mad["MAD (Test)"] = list_mad_testing;

pd_mad['Aws ID'] = subdivs;

fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot(111)

#pd_mad.groupby('Aws ID').mean().plot(title='Overall Rainfall in Each Month', ax=ax,fontsize=20)

pd_mad.groupby('Aws ID').mean().plot.bar( width=0.5,title='MAD for Aws ID', ax= ax, fontsize=20)

plt.xticks(rotation = 90)

plt.ylabel('Mean Abs. Difference')

ax.title.set_fontsize(30)

ax.xaxis.label.set_fontsize(20)

ax.yaxis.label.set_fontsize(20)



from keras.models import model_from_json
# func to save and load file to reduce processing time

import pickle

def save(dictionary, name):

    with open(name, 'wb') as fp:

        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)



# func to load:

def load(kernel2):

    with open(kernel2, 'rb') as fp:

        data = pickle.load(fp)

        return data
# save(dict1,"dict_rainfallFinal")
from keras.layers import SimpleRNN,Dense,LSTM,Dropout,Activation,BatchNormalization

from keras.models import Sequential

from keras import optimizers
model=Sequential()

look_back=10
# Random forest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2, random_state=0)

clf.fit(X_train, y_train)

clf.predict(X_test)

clf.score(X_test, y_test)
#splitting training and testing data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#SVM

from sklearn import svm

from sklearn.datasets import make_classification

clf = svm.SVC()

X_train, y_train = make_classification()

X_test, y_test = make_classification()

clf.fit(X_train, y_train)
clf.predict(X_test)
model.add(SimpleRNN(1000,input_shape=(look_back,12)))

model.add(Dense(12))

model.add(Dropout(0.2,input_shape=(look_back,)))

model.add(Dense(1000))

model.add(Dense(12))

sgd = optimizers.rmsprop(lr=0.005)

model.compile(loss='mean_absolute_error',optimizer='sgd',metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=5,batch_size=4)
ypred=model.predict(X_test)
model.evaluate(X_test[:2],y_test)
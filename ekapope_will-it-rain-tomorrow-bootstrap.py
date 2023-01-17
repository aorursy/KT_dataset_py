#Load the csv file as data frame.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/weatherAUS.csv')

print('Size of weather data frame is :',df.shape)

#Let us see how our data looks like!

df[0:5]
# We see there are some columns with null values. 

# Before we start pre-processing, let's find out which of the columns have maximum null values

df.count().sort_values()
# As we can see the first four columns have less than 60% data, we can ignore these four columns

# We don't need the location column because 

# we are going to find if it will rain in Australia(not location specific)

# We are going to drop the date column too.

# We need to remove RISK_MM because we want to predict 'RainTomorrow' and RISK_MM can leak some info to our model

df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Location','RISK_MM','Date'],axis=1)

df.shape
#Let us get rid of all null values in df

df = df.dropna(how='any')

df.shape
#its time to remove the outliers in our data - we are using Z-score to detect and remove the outliers.

from scipy import stats

z = np.abs(stats.zscore(df._get_numeric_data()))

print(z)

df= df[(z < 3).all(axis=1)]

print(df.shape)
#Lets deal with the categorical cloumns now

# simply change yes/no to 1/0 for RainToday and RainTomorrow

df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)



#See unique values and convert them to int using pd.getDummies()

categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

for col in categorical_columns:

    print(np.unique(df[col]))

# transform the categorical columns

df = pd.get_dummies(df, columns=categorical_columns)

df.iloc[4:9]
#next step is to standardize our data - using MinMaxScaler

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(df)

df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

df.iloc[4:10]
#now that we are done with the pre-processing part, let's see which are the important features for RainTomorrow!

#Using SelectKBest to get the top features!

from sklearn.feature_selection import SelectKBest, chi2

X = df.loc[:,df.columns!='RainTomorrow']

y = df[['RainTomorrow']]

selector = SelectKBest(chi2, k=3)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)]) #top 3 columns
#Let's get hold of the important features as assign them as X

df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]

X = df[['Humidity3pm']] # let's use only one feature Humidity3pm

y = df[['RainTomorrow']]
#Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import time



t0=time.time()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.5)

clf_dt = DecisionTreeClassifier(random_state=0)

clf_dt.fit(X_train,y_train)

y_pred = clf_dt.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
# random samples for bootstraping

print(len(X_train))

print(X_train.head())

# run one df for bootstrap sample, with replacement

df = X_train.sample(n=len(X_train), replace= True, random_state = 0)

print(len(df))

print(df.head())
df.index.values.astype(int)
# put in to for loop (1000 bootstrap samples from X_train)

x_train_list = []

y_train_list = []

ind_list = []

for i in range(1000):

    x_df = X_train.sample(n=len(X_train), replace=True, random_state = i)

    x_train_list.append(x_df)

    y_df = y_train.sample(n=len(y_train), replace=True, random_state = i)

    y_train_list.append(y_df)

    ind_list.append(x_df.index.values.astype(int))

print(len(x_train_list))
# check if the samples are from the same index rows

print(x_train_list[0].head())

print(y_train_list[0].head())
y_pred_list = []

score_list = []

for i in range(1000): 

    clf_dt = DecisionTreeClassifier(random_state=0)

    clf_dt.fit(x_train_list[i],y_train_list[i])

    y_pred = clf_dt.predict(X_test)

    y_pred_list.append(y_pred)

    score = accuracy_score(y_test,y_pred)

    score_list.append(score)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
avg_pred_list = np.mean(y_pred_list, axis=0)

print(avg_pred_list[0:100])

# change to binary 0/1 

binary_list = []

for i in range(len(avg_pred_list)):

    if avg_pred_list[i] > 0.5:  binary_list.append(1)

    else: binary_list.append(0)

print(binary_list[0:100])
score = accuracy_score(y_test,binary_list)

print(score)

# Accuracy (from single train/test run) : 0.8359290985278303
(0.8367263692661401-0.8359290985278303)*100
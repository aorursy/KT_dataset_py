#Load the csv file as data frame.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/weatherAUS.csv')

print('Size of weather data frame is :',df.shape)

#Let us see how our data looks like!

df[0:5]

# taking only 0.1 % of total data

df=df.sample(frac=0.1)



# visualising our dataframe df



import  matplotlib.pyplot as plt

plt.scatter(df["Sunshine"],df["Evaporation"])


# dropping the columns

df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am',

                      'Location','RISK_MM','Date'],axis=1)



df.shape



# reamoving all null values from the data set

df=df.dropna(how ="any")



# there are some columns with null values.

df.count().sort_values()

df.isnull().sum()



# remove the outliers in our data - we are using Z-score to detect

# and remove the outliers.

import numpy as np

from scipy import stats



z=np.abs(stats.zscore(df._get_numeric_data()))



df=df[(z<3).all(axis=1)]



#deal with the categorical cloumns , ie; 0/1

df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)



categorical_columns = ['WindGustDir', 'WindDir3pm', 'WindDir9am']

for col in categorical_columns:

    print(np.unique(df[col]))



df=pd.get_dummies(df,columns=categorical_columns)



df.iloc[4:9]

# Lets standardize our data - using MinMaxScaler



from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()

scaler.fit(df)

df = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)

df.iloc[4:10]

#                 Feature Selection



# let's see which are the important features for RainTomorrow using SelectKBest to get the top features



from sklearn.feature_selection import SelectKBest, chi2

X = df.loc[:,df.columns!='RainTomorrow']

y = df[['RainTomorrow']]

selector = SelectKBest(chi2, k=3)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)])

#Let's get hold of the important features as assign them as X



df = df[['Humidity3pm','Rainfall','RainToday','RainTomorrow']]

X = df[['Humidity3pm']].values # let's use only one feature Humidity3pm

y = df[['RainTomorrow']].values



# logistic regression



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import time



t0=time.time()



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)

clf_logreg = LogisticRegression()

clf_logreg.fit(X_train,y_train)

y_pred = clf_logreg.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
# Decision Tree Classifier



from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

t0=time.time()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

clf_dt = DecisionTreeClassifier(random_state=0)

clf_dt.fit(X_train,y_train)

y_pred = clf_dt.predict(X_test)

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)

print('Time taken :' , time.time()-t0)
# Submission of file



submit = pd.DataFrame({"Predicted Weather":y_pred})

submit.to_csv('submission.csv', index=False)
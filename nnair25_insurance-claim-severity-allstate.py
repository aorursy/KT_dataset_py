# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Header Files 

import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

#Load Dataset

DF= pd.read_csv("../input/train.csv")

DF_Test = pd.read_csv("../input/test.csv")



#To show all the columns and rows

pd.set_option('display.max_rows',None)

pd.set_option('display.max_columns', None)



#Take a look at the dataset

print(DF.head(5))



#Storing the ID from test for future prediction

ID= DF_Test['id']

#Dropping ID because it is meaningless

DF.drop('id',axis=1, inplace=True)

DF_Test.drop('id',axis=1, inplace=True)

#For numerical/continous values

print("Cont. Features")

print("-"*75)

print(DF.describe())



#For categorical values

print("Cat. Features")

print("-"*75)

print(DF.describe(include=['O']))
#Taking only numerical values

size=15

split=116

ContDF=DF.iloc[:,split:]



#Name of all columns

Col=ContDF.columns



#Plotting violin plot for all columns

n_rows=5

n_columns=3



for i in range(n_rows):

    fg,ax = plt.subplots(nrows=1, ncols=n_columns,figsize=(12,8))

    for j in range(n_columns):

        sns.violinplot(y=Col[i*n_columns+j], data=ContDF,ax=ax[j])

        
DF['loss']=np.log1p(DF['loss'])

#Let's visualise the new plot

sns.violinplot(data=DF, y='loss')

plt.show()



#PLOT shows that skew has been corrected to a large extent


CorrMatrix= ContDF.corr().abs()



#Heatmap

plt.subplots(figsize=(13, 9))

sns.heatmap(CorrMatrix,annot=True)

sns.heatmap(CorrMatrix, mask=CorrMatrix < 1, cbar=False)

plt.show()

labellist = []

Col= DF.columns

for i in range(0,split):

    train = DF[Col[i]].unique()

    test = DF[Col[i]].unique()

    labellist.append(list(set(train) | set(test)))    







#Hot encoding all categorical attributes

categ = []

for i in range(0, split):

    #Label encode

    label_encoder = LabelEncoder()

    label_encoder.fit(labellist[i])

    feature = label_encoder.transform(DF.iloc[:,i])

    feature = feature.reshape(DF.shape[0], 1)

    #One hot encode

    onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labellist[i]))

    feature = onehot_encoder.fit_transform(feature)

    categ.append(feature)



# Make a nd.numpyarray

encoded_categ = np.column_stack(categ)







#Combine encoded attributes with continuous attributes

DF_encoded = np.concatenate((encoded_categ,DF.iloc[:,split:].values),axis=1)

#number of rows and columns

r, c = DF_encoded.shape



#create an array which has indexes of columns

i_cols = []

for i in range(0,c-1):

    i_cols.append(i)



#y is the target variable, X is the remaining  data

X = DF_encoded[:,0:(c-1)]

y = DF_encoded[:,(c-1)]





X_train, X_test, y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=7)
model = LinearRegression(n_jobs=-1)

#Accuracy of the model 

model.fit(X_train,y_train)

result = mean_absolute_error(np.expm1(Y_test), np.expm1(model.predict(X_test)))



model = XGBRegressor(n_estimators=1000,seed=7)

#Accuracy of the model 

model.fit(X_train,y_train)

result = mean_absolute_error(np.expm1(Y_test), np.expm1(model.predict(X_test)))

            



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import re

import category_encoders as ce
df = pd.read_csv('../input/BlackFriday.csv')
df.head()
df.tail()
df.info()
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.countplot( x = 'Gender' , data = df)
labels = ['City_A' , 'City_B', 'City_C']

sizes = [ df['City_Category'].value_counts()[2], df['City_Category'].value_counts()[0], df['City_Category'].value_counts()[1]]

explode = (0, 0.1, 0) 



fig1, ax1 = plt.subplots(figsize=(12,7))

ax1.pie(sizes , labels = labels , explode = explode, autopct = '%1.1f%%' , shadow = True)

ax1.axis('equal')

plt.show()
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.countplot( x = 'Age' , data = df)
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.countplot( x = 'Occupation' , data = df)
labels = [ 1 ,2 ,3 , '4+' , 0]

stay_count = df['Stay_In_Current_City_Years'].value_counts()

sizes = [ stay_count[0] , stay_count[1] , stay_count[2] , stay_count[3] , stay_count[4] ]

explode = (0.1 , 0 , 0 , 0 , 0.1)



fig1 , ax1 = plt.subplots(figsize = (12 ,7))

ax1.pie( sizes , labels = labels , explode = explode , autopct = '%1.1f%%' , shadow = True)

ax1.axis('equal')

plt.show()
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.countplot( x = 'Marital_Status' , data = df)
fig1, ax1 = plt.subplots(figsize=(12,7))

sns.countplot(df['City_Category'],hue=df['Age'])
fig1 , ax1 = plt.subplots(figsize = (12,7))

sns.boxplot('Age' , 'Purchase' , data = df)

plt.show()
fig1 , ax1 = plt.subplots(figsize = (12,7))

plt.hist( 'Purchase' , data = df)

plt.show()
pattern = re.compile('\d*\+')



def stay_in_city(row , pattern):

    stay = row['Stay_In_Current_City_Years']

    

    if bool(pattern.match(stay)):

        stay = stay.replace("+","")

        return stay

    else:

        return stay

    

df['Stay_In_Current_City_Years'] = df.apply( stay_in_city , axis = 1 , pattern = pattern )
df[['Product_Category_1' , 'Product_Category_2' , 'Product_Category_3']] = df[['Product_Category_1' , 'Product_Category_2' , 'Product_Category_3']].fillna(0)
df.head()
df.tail()
df.info()
#Creating the dataset copies

dataset = df

#dataset = dataset.drop( columns = 'Unnamed: 0')

df = dataset.copy()
df_dummy = df.iloc[:, 2:].values

pd.DataFrame(df_dummy).head()
#Encoding categorical Data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

df_dummy[:, 0] = labelencoder_X_1.fit_transform(df_dummy[:, 0])

df_dummy[:, 1] = labelencoder_X_1.fit_transform(df_dummy[:, 1])

df_dummy[:, 3] = labelencoder_X_1.fit_transform(df_dummy[:, 3])

#Creating Dummy Variables for City Categories

onehotencoder = OneHotEncoder(categorical_features = [3])

df_dummy = onehotencoder.fit_transform(df_dummy).toarray()



#Removing Dummy variable trap

df_dummy = df_dummy[:, 1:]
pd.DataFrame(df_dummy).head()
#Binary Encoding

encoder = ce.BinaryEncoder(cols = [4 , 7 , 8 ,9]) 

df_dummy = encoder.fit_transform(df_dummy)
df_dummy.head()
#Seprating Independent and Dependent variables

y = df_dummy.iloc[: , 29].values

X = df_dummy.iloc[:, 0:29].values
#Seprating Dataset into Test set and Training set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Fitting Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)

regressor.fit(X_train, y_train)
# Predicting a new result

y_pred = regressor.predict(X_test)
data = { 'Actual_Purchase' : y_test , 'Predicted_Purchase' : y_pred }

pd.DataFrame(data).head(10)
data = { 'Actual_Purchase' : y_test , 'Predicted_Purchase' : y_pred }

pd.DataFrame(data).tail(10)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5)
print ( "Accuracies :"+str(accuracies)+"\nMean_accuracies :"+str(accuracies.mean())+"\nStandard_deviation :"+str(accuracies.std()))
from sklearn.metrics import mean_squared_error , r2_score

print ("RMSE value :"+str(np.sqrt(mean_squared_error(y_test, y_pred))))

print ("R2 Score :"+str(r2_score(y_test , y_pred)))
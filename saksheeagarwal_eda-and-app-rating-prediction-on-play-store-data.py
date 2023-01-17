# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Importing required libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline

import seaborn as sns
#Reading the data

apps=pd.read_csv('../input/google-play-store-apps/googleplaystore.csv', engine='python')

apps.head()
apps.info()
apps['Category'].value_counts()
#Checking for null values using a heatmap

sns.heatmap(apps.isna())
#Imputing the null values in the rating column with the median of that column

apps['Rating'] = apps['Rating'].fillna(apps['Rating'].median())
#Dropping the rest of the nulls and duplicates

apps.dropna(inplace = True) 

apps.drop_duplicates(inplace=True) 
#Converting Last Updated into a datetime type value

apps['Last Updated']=pd.to_datetime(apps['Last Updated'])

apps['before update']=apps['Last Updated'].max()-apps['Last Updated']
#Cleaning Installs for unnecessary characters

apps['Installs']=apps['Installs'].str.replace(',','').str.replace('+','').astype(int)

apps['Installs']
#Converting Reviews to integer

apps['Reviews']=apps['Reviews'].astype(int)
apps['Size']=apps['Size'].str.replace('M','e+6').str.replace('k','e+3').str.replace('Varies with device','0').astype('float')

apps['Price']=apps['Price'].str.replace('$','').astype('float')
apps.info()
#App distribution by category

plt.figure(figsize=(40,12))

apps['Category'].value_counts().plot(kind='pie')
#Plotting the numerical values to find correlations

sns.pairplot(apps, hue='Type')
#Average Rating of the Apps

apps['Rating'].plot(kind='hist', bins=20)
#Content Rating Distribution

plt.figure(figsize=(20,8))

apps['Content Rating'].value_counts().plot(kind='pie')
#Most Installed Category => Communication

plt.figure(figsize=(40,20))

sns.barplot(x='Category',y='Installs', data=apps)
#Average Rating by Category

groups = apps.groupby('Category').filter(lambda x: len(x) > 286).reset_index()

categoryrating = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
#Separating Target Column

X=apps[['Category','Reviews','Size','Installs','Type','Price','Content Rating','Genres','before update']]

y=apps['Rating']



#Scaling 'before update' column

scaler=MinMaxScaler()

scaler.fit(X[['before update']])

X[['before update']]=scaler.transform(X[['before update']])

X
#Encoding categorical values as model works with numerical data

encoded_x=pd.get_dummies(X, columns=['Category',"Content Rating","Type","Genres"])

encoded_x
#Splitting the cleaned and encoded data into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(encoded_x,y,test_size = 0.25, random_state = 10)
#Scaling data for better model score

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.transform(X_test)

X_train.shape,X_test.shape
#Training the model

rf= RandomForestRegressor(n_jobs=-1)

rf.fit(X_train,y_train)
#Predictiong the ratings and checking model score

predictions=rf.predict(X_test)

'Training Score:', rf.score(X_train,y_train),'Testing Score:',rf.score(X_test,y_test)
'Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions))
predict_dataframe = pd.DataFrame(data={"Predicted": predictions, "Actual": y_test})

predict_dataframe
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
df = pd.read_csv('../input/weatherAUS.csv')
#Lets have a quick look of dataset

df.info()
#Now lets clean the dataset

#we can see some columns have less than 60% value,now eleminate that columns, they may affect our model

df =df.drop(columns=['Evaporation','Sunshine', 'Cloud9am','Cloud3pm'], axis =1)

df.shape
#We can eleminate some columns which are really need not to predict,weather it will be rain or not rain tomorrow in Austrilia

df =df.drop(columns=['Date','Location'], axis =1)

df.shape
#Now check the null value of dataset and deal with them

df.isnull().sum()
#Lets remove the null value

df = df.dropna(how = 'any')
#Now lets encode the catagorical data

from sklearn.preprocessing import LabelEncoder

labelencoder_df = LabelEncoder()

df['WindGustDir'] = labelencoder_df.fit_transform(df['WindGustDir'])

df['WindDir9am'] = labelencoder_df.fit_transform(df['WindDir9am'])

df['WindDir3pm'] = labelencoder_df.fit_transform(df['WindDir3pm'])

df['RainToday'] = labelencoder_df.fit_transform(df['RainToday'])

df['RainTomorrow'] = labelencoder_df.fit_transform(df['RainTomorrow'])

#Now lets check the dataset

df.head()

df.tail()
import matplotlib.pyplot as plt

import seaborn as sns

corr = df.corr()

plt.figure(figsize = (12,10))

sns.heatmap(corr,

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values,

            annot=True,fmt='.2f',linewidths=0.30)
#Now lets creat ml model

# Now take our matrix to features

x = df.iloc[:, 0 : 17].values

y = df.iloc[:, -1].values


#Spliting the dataset into training and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

x_test = sc_x.fit_transform(x_test)
#Fitting Random Forest to tranning set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 200, random_state = 0)

classifier.fit(x_train, y_train)

#Predicting the test set result

y_pred = classifier.predict(x_test)


#Making the confusion matrix, accuracy score

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Classification :")

print("Accuracy = ", accuracy)

print(cm)

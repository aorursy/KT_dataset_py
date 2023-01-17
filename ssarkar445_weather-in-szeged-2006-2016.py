import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd



import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LinearRegression



import tensorflow as tf

from tensorflow import keras



from sklearn import preprocessing

tf.__version__
weather_df = pd.read_csv('../input/weatherHistory.csv')
weather_df.head()
weather_df.columns

weather_df.shape

weather_df.describe()

weather_df.info()

weather_df.isnull().any()

weather_df.isnull().all()
round(100*(weather_df.isnull().sum()/len(weather_df.index)),2)
weather_df['Precip Type'].value_counts()
weather_df.loc[weather_df['Precip Type'].isnull(),'Precip Type']='rain'
round(100*(weather_df.isnull().sum()/len(weather_df.index)),2)
plt.scatter(x = 'Wind Speed (km/h)',y = 'Wind Bearing (degrees)',data=weather_df)
plt.figure(figsize = (10,10))

plt.subplot()

plt.subplot(2,3,1)

plt.title('Wind Speed (km/h)')

plt.hist(x = 'Wind Speed (km/h)',bins =20,data = weather_df)



plt.subplot(2,3,2)

plt.title('Apparent Temperature (C)')

plt.hist(x = 'Apparent Temperature (C)',bins =20,data = weather_df)



plt.subplot(2,3,3)

plt.title('Humidity')

plt.hist(x = 'Humidity',bins =20,data = weather_df)



plt.subplot(2,3,4)

plt.title('Wind Bearing (degrees)')

plt.hist(x = 'Wind Bearing (degrees)',bins =100,data = weather_df)



plt.subplot(2,3,5)

plt.title('Pressure (millibars)')

plt.hist(x = 'Pressure (millibars)',bins =20,data = weather_df)



plt.subplot(2,3,6)

plt.title('Visibility (km)')

plt.hist(x = 'Visibility (km)',bins =20,data = weather_df)





plt.show()
## Creating corellation metrics

weather_corr = weather_df[list(weather_df.dtypes[weather_df.dtypes!='object'].index)].corr()
sns.heatmap(weather_corr,annot=True)
sns.pairplot(weather_df)
# Imputing binary values in type column 

weather_df.loc[weather_df['Precip Type']=='rain','Precip Type']=1

weather_df.loc[weather_df['Precip Type']=='snow','Precip Type']=0
weather_df_num=weather_df[list(weather_df.dtypes[weather_df.dtypes!='object'].index)]
weather_y = weather_df_num.pop('Temperature (C)')

weather_X = weather_df_num
train_X,test_X,train_y,test_y = train_test_split(weather_X,weather_y,test_size = 0.2,random_state=4)
train_X.head()
model = LinearRegression()

model.fit(train_X,train_y)
prediction = model.predict(test_X)
## Calculating the error 

np.mean((prediction-test_y)**2)
pd.DataFrame({'actual':test_y,

             'prediction':prediction,

             'diff':(test_y-prediction)})
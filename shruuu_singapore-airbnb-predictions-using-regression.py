# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualisation

import matplotlib.pyplot as plt

import seaborn as sns

#for data preprocessing

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

#for selecting the best feature effecting the target variable

from sklearn.feature_selection import SelectKBest,chi2

# for splitting the data into train and test dataset

from sklearn.model_selection import train_test_split

from sklearn.svm import SVR

# regression metrics

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from math import sqrt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset=pd.read_csv('../input/singapore-airbnb/listings.csv')
dataset.head()
dataset.info()
dataset.drop(['id','name','host_name','host_id','last_review'],axis=1,inplace=True)
dataset.describe()
correlation_matrix=dataset.corr()

sns.heatmap(correlation_matrix)
dataset.drop(['reviews_per_month'],axis=1,inplace=True)
dataset.shape
dataset.isnull().sum()
names=['latitude','longitude','price','minimum_nights','number_of_reviews','calculated_host_listings_count','availability_365']

plt.figure(figsize=(10,9))

for i in range(1,8):

    

    plt.subplot(2,4,i)

    fig=dataset.boxplot(column=names[i-1])
plt.figure(figsize=(12,10))

for j in range(1,8):

    plt.subplot(2,4,j)

    sns.distplot(dataset[names[j-1]])
#for latitude

std=np.std(dataset['latitude'])

mean=np.mean(dataset['latitude'])

median=np.median(dataset['latitude'])

outliers=[]

for x in dataset['latitude']:

    zscore=(x-mean)/std

    if zscore>abs(3):

        outliers.append(x)
len(outliers)
dataset_new=dataset.replace(outliers,median)
plt.figure(figsize=(7,5))

plt.subplot(1,2,1)

fig=sns.distplot(dataset['latitude'])

plt.title('before removing outliers')

plt.subplot(1,2,2)

fig2=sns.distplot(dataset_new['latitude'])

plt.title('after removing outliers')
#for longitude

std=np.std(dataset['longitude'])

mean=np.mean(dataset['longitude'])

median=np.median(dataset['longitude'])

outliers=[]

for x in dataset['longitude']:

    zscore=(x-mean)/std

    if -3<zscore>3:

        outliers.append(x)
len(outliers)
#for minimum_nights

q1=dataset['minimum_nights'].quantile(0.25)

q3=dataset['minimum_nights'].quantile(0.75)

outlier=[]

iqr=q3-q1

lower_bound=q1-(1.5*iqr)

upper_bound=q3+(1.5*iqr)

for i in dataset['minimum_nights']:

    if i<lower_bound or i>upper_bound:

            outlier.append(i)        
len(outlier)
plt.figure(figsize=(20,8))

sns.countplot(outlier)
dataset_new=dataset[dataset['minimum_nights']<=365]
plt.figure(figsize=(7,5))

plt.subplot(1,2,1)

sns.boxplot(dataset['minimum_nights'])

plt.title('before removing outliers')

plt.subplot(1,2,2)

sns.boxplot(dataset_new['minimum_nights'])

plt.title('after removing outliers')
#for calculated_host_listings_count

q1=dataset['calculated_host_listings_count'].quantile(0.25)

q3=dataset['calculated_host_listings_count'].quantile(0.75)

outlier=[]

iqr=q3-q1

lower_bound=q1-(1.5*iqr)

upper_bound=q3+(1.5*iqr)

for i in dataset['calculated_host_listings_count']:

    if i<lower_bound or i>upper_bound:

            outlier.append(i)        
len(outlier)
sns.countplot(outlier)
dataset_new['room_type'].unique()
mappings={'Entire home/apt':1,'Private room':2,'Shared room':3}

dataset_new['room_type']=dataset_new['room_type'].map(mappings)
dataset_new.head()
dataset_new['neighbourhood'].unique()
len(dataset_new['neighbourhood'].unique())
import category_encoders as ce

binary=ce.BinaryEncoder(cols=['neighbourhood'])

dataset_new=binary.fit_transform(dataset_new)
dataset_new.head()
x=dataset_new.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14,15]].values

y=dataset_new.iloc[:,12:13].values
dataset['neighbourhood_group'].unique()
# In the ndarray of independent features 'neighbourhood_group' is at 0th position

label=LabelEncoder()

x[:,0]=label.fit_transform(x[:,0])
best_features=SelectKBest(score_func=chi2,k=5)

fit=best_features.fit(x,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(['neighbourhood_group', 'neighbourhood_0', 'neighbourhood_1',

       'neighbourhood_2', 'neighbourhood_3', 'neighbourhood_4',

       'neighbourhood_5', 'neighbourhood_6', 'latitude', 'longitude',

       'room_type','minimum_nights', 'number_of_reviews',

       'calculated_host_listings_count', 'availability_365'])

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']

result=featureScores.nlargest(5,'Score')

print(result)
plt.figure(figsize=(12,5))

sns.barplot(x=result['Specs'],y=result['Score'])
sns.pairplot(dataset)
# scaling the features is necessary to implement svr 

sc_x=StandardScaler()

sc_y=StandardScaler()

x=sc_x.fit_transform(x)

y=sc_y.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
regressor=SVR(kernel='rbf')

regressor.fit(x_train,y_train)

predictions=regressor.predict(x_test)
predict=pd.DataFrame(predictions)

ytest=pd.DataFrame(y_test)

resultant=pd.concat([predict,ytest],axis=1)
resultant.columns=['Predicted_values','Actual_values']
resultant.head()
mae=mean_absolute_error(y_test,predictions)

rmse=sqrt(mean_squared_error(y_test,predictions))
mae
rmse
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns; sns.set(style="ticks", color_codes=True)

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
train=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/train.csv")

test=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-5/test.csv")
#Display the first 100 rows

train.head(100)

#Generates the basic statistics of the dataset

train.describe()

#Check the datatypes of different columns in pandas df

train.dtypes

#Checks for total no of missing values in each column of the datadframe 

train.isnull().sum()

train['County'].fillna(("NAN"), inplace=True)

test['County'].fillna(("NAN"), inplace=True)

train['Province_State'].fillna(("NAN"), inplace=True)

test['Province_State'].fillna(("NAN"), inplace=True)

train.isnull().sum()

test.isnull().sum()



# creating instance of labelencoder in training data

labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column

train['Target_Cat'] = labelencoder.fit_transform(train['Target'])

train['Country_Region_Cat'] = labelencoder.fit_transform(train['Country_Region'])

train['County_Cat'] = labelencoder.fit_transform(train['County'])

train['Province_State_Cat'] = labelencoder.fit_transform(train['Province_State'])



#Converting date to an integer type

#newdate = pd.to_datetime(train['Date'], errors='coerce')

train['Date'] = pd.to_datetime(train['Date'])

#train['Date']= newdate.dt.strftime("%Y%m%d").astype(int)

train['Dayofweek'] = train['Date'].dt.dayofweek

train['Day'] = train['Date'].dt.day

train['Month'] = train['Date'].dt.month

#newdate

train

# creating instance of labelencoder in testing data

labelencoder = LabelEncoder()

# Assigning numerical values and storing in another column

test['Target_Cat'] = labelencoder.fit_transform(test['Target'])

test['Country_Region_Cat'] = labelencoder.fit_transform(test['Country_Region'])

test['County_Cat'] = labelencoder.fit_transform(test['County'])

test['Province_State_Cat'] = labelencoder.fit_transform(test['Province_State'])



#Converting date to an integer type

#newdate = pd.to_datetime(test['Date'], errors='coerce')

#test['Date']= newdate.dt.strftime("%Y%m%d").astype(int)

#test
test['Date'] = pd.to_datetime(test['Date'])

test['Dayofweek'] = test['Date'].dt.dayofweek

test['Day'] = test['Date'].dt.day

test['Month'] = test['Date'].dt.month

#newdate

train
#Exploring the pairwise relationship in a dataset using Seaborn

sns.pairplot(train, height=3)

                # vars=['Population', 'Weight', 'Country_Region_Cat', 'County_Cat', 'Province_State_Cat', 'Dayofweek','Day', 'Month'],dropna = True )

#"Population", "Weight", "TargetValue", "Date", "Target_Cat", "Country_Region_Cat"
#Visualizing the correlation between different variables in the dataset 

plt.title("Heatmap Correlation of 'Covid19' Dataset", fontsize = 10)

sns.heatmap(train.corr(), annot=True, fmt=".2f")

plt.show()
from sklearn.model_selection import train_test_split



# Get features from the Training Dataset

feature_cols = ['Population', 'Weight','Target_Cat', 'Country_Region_Cat', 'Dayofweek','Day', 'Month']

X = train[feature_cols] # Features

y = train['TargetValue'] # Target variable

 



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=20, random_state=0)

regressor.fit(X_train, y_train)
#targets = train['Target'].unique()



#for index in range(0, len(targets)):

    #test['Target'].replace(targets[index], index, inplace=True)

#targets
#Get features for the Test Data

test_feature_cols = ['Population', 'Weight','Target_Cat', 'Country_Region_Cat', 'Dayofweek','Day', 'Month']

testData = test[test_feature_cols]
# predictions

y_pred = regressor.predict(testData)

y_pred

# Set Format

#listPrediction = [int(x) for x in y_pred]

#test.index

#newDF = pd.DataFrame({'Number': testData.index, 'Population': testData['Population'], 'val': listPrediction})

#newDF
#Q05 = newDF.groupby('Number')['val'].quantile(q=0.05).reset_index()

#Q50 = newDF.groupby('Number')['val'].quantile(q=0.5).reset_index()

#Q95 = newDF.groupby('Number')['val'].quantile(q=0.95).reset_index()



#Q05.columns=['Number','0.05']

#Q50.columns=['Number','0.5']

#Q95.columns=['Number','0.95']

#concatDF = pd.concat([Q05,Q50['0.5'],Q95['0.95']],1)

#concatDF['Number'] = concatDF['Number'] + 1

#concatDF.head(10)
#sub = pd.melt(concatDF, id_vars=['Number'], value_vars=['0.05','0.5','0.95'])

#sub['ForecastId_Quantile']=sub['Number'].astype(str)+'_'+sub['variable']

#sub['TargetValue']=sub['value']

#sub=sub[['ForecastId_Quantile','TargetValue']]

#sub.reset_index(drop=True,inplace=True)

#sub.to_csv("submission.csv",index=False)

#sub.head(10)
fid = test['ForecastId']

output = pd.DataFrame({'id':fid,'TargetValue':y_pred})

output
a=output.groupby(['id'])['TargetValue'].quantile(q=0.05).reset_index()

b=output.groupby(['id'])['TargetValue'].quantile(q=0.5).reset_index()

c=output.groupby(['id'])['TargetValue'].quantile(q=0.95).reset_index()
a.columns=['Id','q0.05']

b.columns=['Id','q0.5']

c.columns=['Id','q0.95']

a=pd.concat([a,b['q0.5'],c['q0.95']],1)

a['q0.05']=a['q0.05']

a['q0.5']=a['q0.5']

a['q0.95']=a['q0.95']

a
sub=pd.melt(a, id_vars=['Id'], value_vars=['q0.05','q0.5','q0.95'])

sub['variable']=sub['variable'].str.replace("q","", regex=False)

sub['ForecastId_Quantile']=sub['Id'].astype(str)+'_'+sub['variable']

sub['TargetValue']=sub['value']

sub=sub[['ForecastId_Quantile','TargetValue']]

sub.reset_index(drop=True,inplace=True)

sub.to_csv("submission.csv",index=False)

sub.head()



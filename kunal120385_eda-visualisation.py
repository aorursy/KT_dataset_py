# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
trainDS = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

trainDS.head(5)
trainDS['Province_State'].fillna(trainDS['Country_Region'],inplace=True)

trainDS.head()

trainDS.head()
testDS = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv");

testDS.head()
trainDS = trainDS.rename(columns= {'Province_State':'State','Country_Region':'Country'})
trainDS.head()
testDS = testDS.rename(columns={'Province_State':'State','Country_Region':'Country'})
testDS.head()
trainDS.isnull()
trainDS.info()
trainDS['Date'] = pd.to_datetime(trainDS['Date'], format = '%Y-%m-%d')

testDS['Date']  = pd.to_datetime(testDS['Date'], format = '%Y-%m-%d')
trainDS.info()
testDS.info()
current_date = trainDS['Date'].max()

print(current_date)
trainDS['ConfirmedCases'].sort_values()

trainDS.groupby('Country').ConfirmedCases.value_counts().reset_index(name='Count')
trainDS['total'] = trainDS['ConfirmedCases']+trainDS['Fatalities']
trainDS.head()
trainDS.groupby('Country').total.value_counts().reset_index(name='counts')
countryCount = trainDS.groupby('Country')
print(countryCount.groups)
countryCount['ConfirmedCases'].agg(np.mean)
countryCount.get_group('India')
India_country = countryCount.get_group('India').max()

display(India_country)
China_country = countryCount.get_group('China').max()

display(China_country)
len(countryCount.groups)
initial_data = []

df = pd.DataFrame(initial_data, columns)

df
type(countryCount)
df = pd.DataFrame(columns=['Id','Date','Country', 'ConfirmedCase','Fatalities','Total'])

print('Display Total Cases Country Wise \n\t')

for i, Country in enumerate(countryCount.groups):

    print(i)

    if(countryCount.get_group(Country).max()[6] != 'None'):

        df.loc[i,'Id'] = countryCount.get_group(Country).max()[0]

        df.loc[i,'Date'] = countryCount.get_group(Country).max()[3]

        df.loc[i, 'Country'] = countryCount.get_group(Country).max()[2]

        df.loc[i, 'ConfirmedCase'] = countryCount.get_group(Country).max()[4]

        df.loc[i, 'Fatalities'] = countryCount.get_group(Country).max()[5]

        df.loc[i, 'Total'] = countryCount.get_group(Country).max()[5]+countryCount.get_group(Country).max()[4]

    

    print(countryCount.get_group(Country).max()[0])

    print(countryCount.get_group(Country).max()[1])

    print(countryCount.get_group(Country).max()[2])

    print(countryCount.get_group(Country).max()[3])

    print(countryCount.get_group(Country).max()[4])

    print(countryCount.get_group(Country).max()[5])

    print("\n")
columns
df.sort_values(['Total'], ascending=False)
df.dtypes
df.info()
convert_dict = {'ConfirmedCase': int, 

                'Fatalities': int,

                'Total':int

               } 

  

df = df.astype(convert_dict) 

print(df.dtypes) 
df.head()
df.nlargest(10, 'ConfirmedCase')
df.dtypes
df
from sklearn.datasets import load_svmlight_files

from sklearn.metrics import accuracy_score

from xgboost.sklearn import XGBClassifier
X = df              #Features stored in X 

y = testDS          #Class variable
X
y
import matplotlib.pyplot as plt

from sklearn import  metrics, model_selection

from xgboost.sklearn import XGBClassifier
df['Id'],_ = pd.factorize(df['Id'])

df['Date'],_ = pd.factorize(df['Date'])

df['Country'],_ = pd.factorize(df['Country'])

df['ConfirmedCase'],_ = pd.factorize(df['ConfirmedCase'])

df['Fatalities'],_ = pd.factorize(df['Fatalities'])

df['Total'],_ = pd.factorize(df['Total'])

df.head()
df.dtypes
df.info()
X = df.iloc[:,:-1]

y = df.iloc[:,-1]

print(X)

print(y)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=123)
params = {

    'objective': 'binary:logistic',

    'max_depth': 2,

    'learning_rate': 1.0,

    'silent': 1,

    'n_estimators': 5

}



model = XGBClassifier(**params).fit(X_train, y_train)
# use the model to make predictions with the test data

y_pred = model.predict(X_test)

# how did our model perform?

count_misclassified = (y_test != y_pred).sum()

print('Misclassified samples: {}'.format(count_misclassified))

accuracy = metrics.accuracy_score(y_test, y_pred)

print('Accuracy: {:.2f}'.format(accuracy))
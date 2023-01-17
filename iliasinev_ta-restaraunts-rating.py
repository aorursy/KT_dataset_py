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
import pandas as pd # Like if it was was possible with out it

import re # For string scraping

import numpy as np # Just for NaNs



# Visualization

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
# Setting up some parameters and lib versions for reproducibility 



RANDOM_SEED = 42

CURRENT_DATE = '04/10/2020'



!pip freeze > requirements.txt
df_train = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/main_task.csv')

df_test = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/kaggle_task.csv')



df_train['sample'] = 1 

df_test['sample'] = 0 

df_test['Rating'] = 0 

data = df_test.append(df_train, sort=False).reset_index(drop=True) 
data.info()

data.head()
data['Price Range'].value_counts()
data['Reviews'].iloc[1]
data['Number of Reviews Was NAN'] = data['Number of Reviews'].isna()

data['Number of Reviews'].fillna(0, inplace=True)
data['Price Range'] = data['Price Range'].map({'$':1,'$$ - $$$':2,'$$$$':3}).fillna(2)

# First filling the missing values with 'Unknown' label, labelling as before that initially there was no cuisine style

data['Cuisine Style Was NAN'] = data['Cuisine Style'].isna()

data['Cuisine Style'] = data['Cuisine Style'].fillna("['Unknown']")



# Could have used as 

data['Number of Styles'] = data['Cuisine Style'].str[2:-2].str.split("', '")

data['Number of Styles'] = data['Number of Styles'].apply(lambda x: 0 if x is np.nan else len(x))
data.sample(5)
print(data['City'].value_counts()) # Restaraunts distribution over the cities

print(data['City'].nunique()) # Unique cities
rest_count = data.groupby('City')['Restaurant_id'].count().to_dict()

data['Restaurants in City'] = data['City'].map(rest_count)
data['City ranking'] = data['Ranking'] / data['Restaurants in City']
# Loading a tool to encode cathegories

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer





countries = {

    'London': 'GB',

    'Paris': 'FR',

    'Madrid': 'ES',

    'Barcelona': 'ES',

    'Berlin': 'DE',

    'Milan': 'IT',

    'Rome': 'IT',

    'Prague': 'CZ',

    'Lisbon': 'PT',

    'Vienna': 'AT',

    'Amsterdam': 'NL',

    'Brussels': 'BE',

    'Hamburg': 'DE',

    'Munich': 'DE',

    'Lyon': 'FR',

    'Stockholm': 'SE',

    'Budapest': 'HU',

    'Warsaw': 'PL',

    'Dublin': 'IE',

    'Copenhagen': 'DK',

    'Athens': 'GR',

    'Edinburgh': 'GB',

    'Zurich': 'CH',

    'Oporto': 'PT',

    'Geneva': 'CH',

    'Krakow': 'PL',

    'Oslo': 'NO',

    'Helsinki': 'FI',

    'Bratislava': 'SK',

    'Luxembourg': 'LU',

    'Ljubljana': 'SI',

}



data['Country'] = data['City'].map(countries)



countries_le = LabelEncoder()

countries_le.fit(data['Country'])

data['Country Code'] = countries_le.transform(data['Country'])
cities_le = LabelEncoder()

cities_le.fit(data['City'])

data['City Code'] = cities_le.transform(data['City'])
populations = {

    'London': 8567000, 

    'Paris':  9904000, 

    'Madrid':  5567000,

    'Barcelona':  4920000,

    'Berlin':  3406000,

    'Milan':  2945000,

    'Rome':  3339000,

    'Prague':  1162000,

    'Lisbon':  2812000,

    'Vienna':  2400000,

    'Amsterdam':  1031000,

    'Brussels':  1743000,

    'Hamburg':  1757000,

    'Munich':  1275000,

    'Lyon':  1423000,

    'Stockholm':  1264000,

    'Budapest':  1679000,

    'Warsaw':  1707000,

    'Dublin':  1059000,

    'Copenhagen':  1085000,

    'Athens':  3242000,

    'Edinburgh':  504966,

    'Zurich':  1108000,

    'Oporto':  1337000,

    'Geneva':  1240000,

    'Krakow':  756000,

    'Oslo':  835000,

    'Helsinki':  1115000,

    'Bratislava':  423737,

    'Luxembourg':  107260,

    'Ljubljana':  314807,

}



data['Population'] = data['City'].map(populations)
tourists = {

    'London':  71.6 ,

    'Paris': 52.55,

    'Madrid': 19.83,

    'Barcelona': 19.29,

    'Berlin': 32.87,

    'Milan': 12.29,

    'Rome': 28.55,

    'Prague': 18.25,

    'Lisbon': 10.76,

    'Vienna': 17.41,

    'Amsterdam': 16.94,

    'Brussels': 8.8,

    'Hamburg': 14.53,

    'Munich': 17.12,

    'Lyon': 6,

    'Stockholm': 14.59,

    'Budapest': 31,

    'Warsaw': 4.6,

    'Dublin': 11.2,

    'Copenhagen': 659,

    'Athens': 5.7,

    'Edinburgh': 2.4,

    'Zurich': 5.7,

    'Oporto': 1.6,

    'Geneva': 3.2,

    'Krakow':  13.5,

    'Oslo': 7.5,

    'Helsinki': 2.4,

    'Bratislava': 1.3,

    'Luxembourg': 1.1,

    'Ljubljana': 1.3,

}



data['Tourist visits'] = data['City'].map(tourists)
data['Review Date'] = data['Reviews'].str[2:-2].str.split("\], \[").str[1].replace('\'', '', regex=True)



# New attribute indicating if no text reviews are available

data['Is_Reviewed'] = data['Review Date'].apply(lambda x: 0 if x == '' else 1)
from datetime import datetime as dt

from datetime import date



# Filling missing values and forming lists in the coulmn

data['Review Date'] = data['Review Date'].replace('', np.nan)

data['Review Date'] = data['Review Date'].str.split(", ")
# Adding two attributes described above 



def datediff (dateslist):

    return abs(max( [dt.strptime(d, '%m/%d/%Y') for d in dateslist]) 

               - min( [dt.strptime(d, '%m/%d/%Y') for d in dateslist])).days



def dayspassed (dateslist):

    return abs(max( [dt.strptime(d, '%m/%d/%Y') for d in dateslist])

              - dt.strptime(CURRENT_DATE, '%m/%d/%Y')).days

    

data['Days Between'] = data['Review Date'][~data['Review Date'].isna()].apply(lambda d: datediff(d))

data['Days Since'] = data['Review Date'][~data['Review Date'].isna()].apply(lambda d: dayspassed(d))
data['Numeric ID'] = data['ID_TA'].apply(lambda id_ta: int(id_ta[1:]))
print('Loswest ID: ', data['Numeric ID'].min(), '\n',

'Highest ID: ', data['Numeric ID'].max())
plt.xlabel('Numeric ID')

plt.ylabel('Count')

plt.title('Empty \'Reviews\'')

data[data['Review Date'].isna()]['Numeric ID'].hist(bins=100, range=(500000, 14000000))
plt.xlabel('Numeric ID')

plt.ylabel('Days')

data[data['Days Between']>0]['Numeric ID'].hist(bins=100, range=(500000, 14000000))
plt.xlabel('Numeric ID')

plt.ylabel('Days')

data[data['Days Since']>0]['Numeric ID'].hist(bins=100, range=(500000, 14000000))
# Filling missing values in both Days columns with zeros



data['Days Between'] = data['Days Between'].fillna(0)

data['Days Since'] = data['Days Since'].fillna(0)
# Forming a separate dataframe with dummy variables and joining it to the maiabsn dataset



styles = data['Cuisine Style'].str[2:-2].str.split("', '").apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0, downcast='infer')

data = data.join(styles)
data.info(verbose=True)
# Selecting object columns and numeric columns

object_columns = [c for c in data.columns if data[c].dtypes == 'object']

numeric_columns = [c for c in data.columns if data[c].dtypes != 'object']



# All columns will not be displayed - a small workaround to see if there are any missing values left :)

sum(data[numeric_columns].isna().sum().to_list())
# Removing object columns

data.drop(object_columns, axis = 1, inplace=True)
data.sample(10)
# Let's separate test data

train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # target

X = train_data.drop(['Rating'], axis=1)
# To test the model before sending the prediction isolating 25%



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
# Model settings from baseline

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Model training

model.fit(X_train, y_train)
# Ratings are listed with step size 0.5 - lets round the predicted data



def rating_round(x, base=0.5):

    return base * round(x/base)



def predict(ds):

    return np.array([rating_round(x) for x in model.predict(ds)])



y_pred = predict(X_test)
# Let's see how good the model is

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# So which attributes worked the best?

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data = test_data.drop(['Rating'], axis=1)
predict_submission = predict(test_data)
sample_submission = pd.read_csv('/kaggle/input/sf-dst-restaurant-rating/sample_submission.csv')

sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)
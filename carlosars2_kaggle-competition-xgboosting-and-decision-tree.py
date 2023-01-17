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
import pandas as pd
import numpy as np
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
countries = pd.read_csv ('/kaggle/input/countries/countries.csv')
countries
countries.shape
train = pd.read_csv ('/kaggle/input/train-users/train_users_2.csv')
train
train.shape
test = pd.read_csv ("/kaggle/input/test-users/test_users.csv")
test
test.shape
df = pd.concat([train, test])
df.shape
df.country_destination.unique()
df.age.mean()
df.age.median()
df.age.min()
df.age.max()
df.age.describe()
y = df.age
x = df.country_destination
fig = sns.boxplot(x,y,)
fig.axis(ymin=0, ymax=2200);
bins = np.arange(0,100,10)
plt.hist (df.age, bins);
pd.set_option('display.max_rows',420)
df.sort_values(by=['age'], ascending = False)
convert = {'age': object}  ### objeto dicionário contendo a mudança
df = df.astype (convert) #### Realizando a alteração do tipo da coluna age para que ela seja objeto.
df.dtypes
#### Making the change of Age between 1900 and 2014 to the age median value '34':
df['age'].replace ({2014:34}, inplace = True)
df['age'].replace ({2013:34}, inplace = True)
df['age'].replace ({2012:34}, inplace = True)
df['age'].replace ({2011:34}, inplace = True)
df['age'].replace ({2010:34}, inplace = True)
df['age'].replace ({2008:34}, inplace = True)
df['age'].replace ({1995:34}, inplace = True)
df['age'].replace ({1953:34}, inplace = True)
df['age'].replace ({1952:34}, inplace = True)
df['age'].replace ({1947:34}, inplace = True)
df['age'].replace ({1949:34}, inplace = True)
df['age'].replace ({1942:34}, inplace = True)
df['age'].replace ({1938:34}, inplace = True)
df['age'].replace ({1936:34}, inplace = True)
df['age'].replace ({1931:34}, inplace = True)
df['age'].replace ({1929:34}, inplace = True)
df['age'].replace ({1928:34}, inplace = True)
df['age'].replace ({1927:34}, inplace = True)
df['age'].replace ({1926:34}, inplace = True)
df['age'].replace ({1925:34}, inplace = True)
df['age'].replace ({1924:34}, inplace = True)
df['age'].replace ({1923:34}, inplace = True)
df['age'].replace ({1922:34}, inplace = True)
df['age'].replace ({1921:34}, inplace = True)
df['age'].replace ({1920:34}, inplace = True)
df['age'].replace ({1919:34}, inplace = True)
df['age'].replace ({1935:34}, inplace = True)
df['age'].replace ({1933:34}, inplace = True)
df['age'].replace ({1932:34}, inplace = True)

df['age'].replace ({2002:34}, inplace = True)
df['age'].replace ({2001:34}, inplace = True)
df['age'].replace ({2000:34}, inplace = True)
df['age'].replace ({1968:34}, inplace = True)
df['age'].replace ({1954:34}, inplace = True)
df['age'].replace ({1951:34}, inplace = True)
df['age'].replace ({1948:34}, inplace = True)
df['age'].replace ({1947:34}, inplace = True)
df['age'].replace ({1945:34}, inplace = True)
df['age'].replace ({1944:34}, inplace = True)
df['age'].replace ({1941:34}, inplace = True)
df['age'].replace ({1940:34}, inplace = True)
df['age'].replace ({1939:34}, inplace = True)
df['age'].replace ({1938:34}, inplace = True)
df['age'].replace ({1937:34}, inplace = True)
df['age'].replace ({1935:34}, inplace = True)
df['age'].replace ({1934:34}, inplace = True)
df['age'].replace ({1933:34}, inplace = True)
df['age'].replace ({1931:34}, inplace = True)
df['age'].replace ({1930:34}, inplace = True)
df['age'].replace ({1928:34}, inplace = True)
df['age'].replace ({1927:34}, inplace = True)
df['age'].replace ({1926:34}, inplace = True)
df['age'].replace ({1925:34}, inplace = True)
df['age'].replace ({1924:34}, inplace = True)
df['age'].replace ({1923:34}, inplace = True)
df['age'].replace ({1922:34}, inplace = True)
df['age'].replace ({1920:34}, inplace = True)
df.sort_values(by=['age'], ascending = False)
bins = np.arange(0,160,10)
plt.hist (df.age, bins);
y = df.age
x = df.country_destination
fig = sns.boxplot(x,y,)
fig.axis(ymin=0, ymax=150);
df.age.mean() #### It has decreased from 49 to 37
df.age.describe()
sns.countplot(data=df, x = 'gender', hue = 'gender');
gender = DataFrame (df['gender'].value_counts())
gender
###### Age Average per gender
df.groupby('gender')[['age']].mean()
sns.countplot(data=df, x = 'signup_method', hue = 'signup_method');
 ###### Age Average per signup_method
df.groupby('signup_method')[['age']].mean()
signup_flow = DataFrame (df['signup_flow'].value_counts())
signup_flow

 ###### Age Average per signup_flow
df.groupby('signup_flow')[['age']].mean()
language = DataFrame (df['language'].value_counts())
language
affiliate = DataFrame (df['affiliate_provider'].value_counts())
affiliate
###### Age Average per affiliate_provider
df.groupby('affiliate_provider')[['age']].mean()
first_browser = DataFrame (df['first_browser'].value_counts())
first_browser
sns.countplot(data=df, x = 'first_device_type', hue = 'first_device_type');
affiliate = DataFrame (df['first_device_type'].value_counts())
affiliate
###### Age Average per device
df.groupby('first_device_type')[['age']].mean()
sns.countplot(data=df, x = 'country_destination', hue = 'country_destination');
country_destination = DataFrame (df['country_destination'].value_counts())
country_destination
###### Age Average per destination
df.groupby('country_destination')[['age']].mean()
###### Gender Average per destination
df.groupby(['country_destination','gender'])['gender'].count()
###### Gender Average per destination
lan_df = DataFrame(df.groupby(['language','country_destination'])['language'].count())
pd.set_option('display.max_rows',158)

lan_df
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(
ascending=False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(20)
df.age.mean()
df['age'].fillna(df['age'].mean(), inplace = True)
df.first_affiliate_tracked.unique()
sns.countplot(data=df, x = 'first_affiliate_tracked', hue = 'first_affiliate_tracked');
df['first_affiliate_tracked'].fillna('untracked', inplace = True)
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(
ascending=False)
missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])
missing_data.head(20)
#### Inserting a Date Reference in the dataset
df['date_reference'] = '2019-01-01'
### Converting the date to a format which can allow us to make date calculations
df['date_account_created'] = pd.to_datetime(df['date_account_created'], format = '%Y/%m/%d')
### Converting the date to a format which can allow us to make date calculations
df['date_reference'] = pd.to_datetime(df['date_reference'], format = '%Y/%m/%d')
#### Creating the columns n_of_days 
df['number_of_days_account'] = df['date_reference'] - df['date_account_created'] 
df['n_of_days'] = df.number_of_days_account.astype(str).str[:4].astype(int)

df.n_of_days.describe()
bins = np.arange(0,4000,100)
plt.hist (df.n_of_days, bins);

df.drop(['id','date_reference','date_account_created','date_first_booking','timestamp_first_active','number_of_days_account'], axis=1, inplace= True) 
df
df.dtypes
convert = {'signup_flow': object}   
df = df.astype (convert)  
df.dtypes
#### Retirando a Variável Target do Dataset e colocando em um dataframe separado:
target = df['country_destination']
df.drop(['country_destination'],axis=1, inplace=True)

target.shape
df.dtypes
df_new = pd.get_dummies(df)
pd.set_option('display.max_rows',178)

df_new.dtypes
df_new.shape

##### Seperating the numerical variables
train_numerical = DataFrame(df_new, columns =['age','n_of_days'])
train_numerical_backup = DataFrame(df_new, columns =['age','n_of_days'])
train_numerical.shape
train_numerical.describe()
df_new["age"]=((df_new["age"]-df_new["age"].min())/
                        (df_new["age"].max()-df_new["age"].min()))*1
df_new["n_of_days"]=((df_new["n_of_days"]-df_new["n_of_days"].min())/
                        (df_new["n_of_days"].max()-df_new["n_of_days"].min()))*1
df_new['age'].describe()
df_new['n_of_days'].describe()
df_new
df_train = df_new[:213451]
df_test_kaggle = df_new[-62096:]
df_train.shape
df_test_kaggle.shape
target = target[:213451]
target.shape
##### Splitting the Train dataset to run the datamodel

X_treino, testeData, Y_treino, testeLabels = train_test_split(df_train, target, test_size = 0.30, random_state = 101)
X_treino.shape
testeData.shape
Y_treino.shape
testeLabels.shape
# Model creation

clf = DecisionTreeClassifier()
print(clf)
#### fitting the model
modelo = clf.fit(X_treino, Y_treino)
### make the previsions
previsoes = modelo.predict(testeData)
#### Accuracy
print (accuracy_score(testeLabels, previsoes))
### Importing the packages
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
### Creating the model
clf = XGBClassifier()
#### fitting the model
clf.fit(X_treino, Y_treino)
#### previsions
pred = clf.predict(testeData)
#### Accuracy
print("Accuracy of model is: ", accuracy_score(testeLabels, pred))

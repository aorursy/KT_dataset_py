# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
unclean_focus = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/unclean focus.csv')

cclass = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/cclass.csv')

bmw = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/bmw.csv')

merc = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/merc.csv')

hyundi = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/hyundi.csv')

focus = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/focus.csv')

vauxhall = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/vauxhall.csv')

unclean_cclass = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/unclean cclass.csv')

vw = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv')

audi = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/audi.csv')

ford= pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/ford.csv')

skoda = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/skoda.csv')

toyota = pd.read_csv('/kaggle/input/used-car-dataset-ford-and-mercedes/toyota.csv')
unclean_focus.head(5)
focus.head()
audi.head(5)
bmw.head(5)
vw.head(5)
cclass.head(5)
cclass['csv'] = 'cclass'

bmw['csv'] = 'bmw'

merc['csv'] = 'merc'

hyundi['csv'] = 'hyundi'

focus['csv'] = 'focus'

vauxhall['csv'] = 'vauxhall'

vw['csv'] = 'vw'

audi['csv'] = 'audi'

ford['csv'] = 'ford'

skoda['csv'] = 'skoda'

toyota['csv'] = 'toyota'
df_original = cclass.append([bmw, merc, hyundi, focus, vauxhall, vw, audi, ford, skoda, toyota], ignore_index=False, verify_integrity=False, sort=False)

df = cclass.append([bmw, merc, hyundi, focus, vauxhall, vw, audi, ford, skoda, toyota], ignore_index=False, verify_integrity=False, sort=False)
df
df
df[df['tax(£)'].notnull() == True]
df[(df['tax(£)'].notnull() == True) & (df['tax'].notnull()==True)]
df['tax(£)'].fillna(value=0, inplace=True)

df['tax'].fillna(value=0, inplace=True)

df['tax'] = df['tax(£)'] + df['tax']

df.drop(labels='tax(£)', axis=1, inplace=True)

df
df.info()
df.describe()
def univariate_plots(column, data=df):

    if data[column].dtype not in ['int64', 'float64']:

        f, axes = plt.subplots(1,1,figsize=(15,5))

        sns.countplot(x=column, data = data)

        plt.xticks(rotation=90)

        plt.suptitle(column,fontsize=20)

        plt.show()

    else:

        g = sns.FacetGrid(data, margin_titles=True, aspect=4, height=3)

        g.map(plt.hist,column,bins=100)

        plt.show()

    plt.show()
for column in df.columns:

    univariate_plots(column)
df['model'].value_counts()
(df[(df['year']>2020) | (df['year']<1990)]).describe()
(df[(df['year']<2020) | (df['year']>1990)]).describe()
def convert_year(year):

    if year > 2020 or year < 1990:

        year = 2017 

    else:

        year = year

    

    return year
df['year'] = df['year'].apply(convert_year)

df[(df['year']>2020) | (df['year']<1990)]
df[df['transmission'] == 'Other']
for model in df[df['transmission'] == 'Other']['model'].unique():

    print(model)

    print(df[df['model'] == model]['transmission'].value_counts())
def convert_transmission(transmission, model = df['model']):

    if transmission not in ['Manual','Semi-Auto','Automatic']:

        transmission = df[df['model'] == model]['transmission'].value_counts().reset_index()['index'][0]

    else:

        transmission = transmission

    

    return transmission
df['transmission'] = df['transmission'].apply(convert_transmission)
sns.pairplot(df)
categorical_columns = []



for column in df_original.columns:

    if df_original[column].dtype == 'object':

        categorical_columns.append(column)

        

df_original = pd.get_dummies(df_original,columns=categorical_columns, dtype=int, drop_first=True)

df_original.fillna(0, inplace=True)



y = df_original['price']

X = df_original.drop(labels = ['price'], axis = 1)



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



model = LinearRegression().fit(X_train, y_train)

y_pred = model.predict(X_test)



from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('r2_score:', metrics.r2_score (y_test, y_pred))
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



categorical_columns = []



for column in df.columns:

    if df[column].dtype == 'object':

        categorical_columns.append(column)

        

df = pd.get_dummies(df,columns=categorical_columns, dtype=int, drop_first=True)

df.fillna(0, inplace=True)



y = df['price']

X = df.drop(labels = ['price'], axis = 1)



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



models = [DecisionTreeRegressor(), LinearRegression(), Ridge(),  Lasso()]



for model in models:

    

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)



    from sklearn import metrics

    print('Model:', model)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    print('r2_score:', metrics.r2_score (y_test, y_pred))

    print('-------------------------------------')
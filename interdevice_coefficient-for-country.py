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
base = pd.read_csv('../input/master.csv')
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
def country(x):

    country = x;

    return country
ct = base['country'].map(country)
ctu = [];

def unico(ct):

    for i in ct:

        if i not in ctu:

            ctu.append(i);
unico(ct)
def geracao(x):

    if x == '5-14 years':

        return 1

    elif x == '15-24 years':

        return 2

    elif x == '25-34 years':

        return 3

    elif x == '35-54 years':

        return 4

    elif x == '55-74 years':

        return 5

    elif x == '75+ years':

        return 6

    else:

        return 0
base['idade'] = base['age'].map(geracao)
def countries(i):

    country = base[base['country'] == i]

    y = country['suicides_no']

    country = country.drop(columns=['year','suicides_no','country', 'sex', 'age', 'country-year',' gdp_for_year ($) ', 'generation'])

    X =  country

    X.fillna(0.0, inplace = True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    lm = LinearRegression()

    lm.fit(X_train,y_train)

    coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

    print(i)

    print(coeff_df)

    print('\n')
for i in ctu:

    countries(i);
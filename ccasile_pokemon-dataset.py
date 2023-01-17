import pandas as pd

pokemon = pd.read_csv("../input/pokemon/pokemon.csv")



pokemon.head()
pokemon.isna().sum()
pokemon.info()
pokemon.describe()
count = 0

for data in pokemon.is_legendary:

    if data == 1:

        count += 1



print(count)
y = pokemon['is_legendary']

X = pokemon

X = X.drop(columns=['is_legendary'])

X = X.drop(columns=['type2'])

X = X.drop(columns=['name'])

X = X.drop(columns=['japanese_name'])

X = X.drop(columns=['abilities'])

X = X.drop(columns=['pokedex_number'])



# This line can be commented out to run all X variables

X = X[['attack', 'sp_attack', 'sp_defense', 'speed', 'weight_kg', 'percentage_male', 'height_m', 'defense', 'base_egg_steps', 'type1']]

X = pokemon[['attack', 'speed', 'type1','weight_kg', 'percentage_male', 'height_m']]

X.head(25)
X.isna().sum()
average_height = pokemon.height_m.mean()

average_weight = pokemon.weight_kg.mean()

num = 50



X.height_m.fillna(average_height, inplace=True)

X.weight_kg.fillna(average_weight, inplace=True)

X.percentage_male.fillna(num, inplace=True)
X.isna().sum()
X = pd.get_dummies(X, drop_first=True)

X.head()
X.corr()
from sklearn import preprocessing



cols = X.columns

x = X.values 

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

X = pd.DataFrame(x_scaled, index=X.index, columns=cols)

X.head()
y.head()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(random_state=0)



logreg.fit(X_train,y_train)
predictions = logreg.predict(X_test)



from sklearn.metrics import accuracy_score



accuracy_score(y_test, predictions)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, predictions)
X.info()
from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from scipy import stats



test = X['attack']



X2 = sm.add_constant(test)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from scipy import stats



test = pokemon[['attack', 'speed']]



X2 = sm.add_constant(test)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

from scipy import stats



test = X[['speed']]

#test = X[['attack', 'speed', 'percentage_male', 'defense', 'sp_defense', 'base_egg_steps' ]]

test = pokemon[['speed', 'base_egg_steps']]



X2 = sm.add_constant(test)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
X_train, X_test, y_train, y_test = train_test_split(test, y, test_size=0.25, random_state=0)



logreg1 = LogisticRegression(random_state=0)



logreg1.fit(X_train,y_train)



predictions = logreg1.predict(X_test)



accuracy_score(y_test, predictions)
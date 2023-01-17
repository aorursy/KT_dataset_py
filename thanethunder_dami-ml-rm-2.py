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
MELBOURNE = pd.read_csv("../input/melb_data.csv")

MELBOURNE.head()
MISSING = MELBOURNE.isnull().sum()

print(MISSING[MISSING > 0])
#Create a new dataset without the columns that have at least one missing

MELBOURNE_MISSING_FREE = MELBOURNE.dropna(axis=1)

print(MELBOURNE_MISSING_FREE.head())

DIMENSIONS = [[MELBOURNE.shape], [MELBOURNE_MISSING_FREE.shape]]

print(DIMENSIONS)



#Sklearn algorithm to fill nan with the mean of the column

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

MEBOURNE_MEAN_IMPUTER = my_imputer.fit_transform(MELBOURNE)

#DON'T PANIC!!!:

#L'errore che viene fuori è dovuto alla presenza di categoriche (stringhe)

#per il momento imputiamo i valori per le sole variabili quantitative
#print(MELBOURNE.dtypes)

#print(MELBOURNE.iloc[0])

#prendiamo solo le variabili numeriche (int64, float64)

MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE = MELBOURNE.select_dtypes(exclude = ["object"])

COLUMNS = MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE.columns

MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE = my_imputer.fit_transform(MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE)

MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE = pd.DataFrame(MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE, columns=COLUMNS)

MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE.head()







#print(MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE.head())

#DIMENSIONS = [[MELBOURNE.shape], [MELBOURNE_MISSING_FREE.shape], [MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE.shape]]

#print(DIMENSIONS)

#print(MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE.columns)

y = MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE.Price

X = MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE.drop("Price", axis=1)

POTENTIAL_REGRESSOR = ["Rooms", "Distance", "YearBuilt"]

#X = MELBOURNE_MEAN_IMPUTER_CATEGORIES_FREE[POTENTIAL_REGRESSOR]

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(X,y)



import statsmodels.api as sm



X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())
# make copy to avoid changing original data (when Imputing)

NEW_MELBOURNE = MELBOURNE.copy()

NEW_MELBOURNE.columns = MELBOURNE.columns



# make new columns indicating what will be imputed

cols_with_missing = (col for col in NEW_MELBOURNE.columns 

                                 if NEW_MELBOURNE[col].isnull().any())

for col in cols_with_missing:

    NEW_MELBOURNE[col + '_was_missing'] = NEW_MELBOURNE[col].isnull()



"""

# Imputation

my_imputer = SimpleImputer()

new_data = pd.DataFrame(my_imputer.fit_transform(new_data))

new_data.columns = original_d

"""

# DON'T PANIC: l'ultimo blocco non va perché sono presenti variabili categoriche e non posso fare l'imputazione numerica della media
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



#riusero la stessa y di sopra. Come X utilizzo tutte le numeriche(elimino le categoriche) che presentano i missing 

melb_X = MELBOURNE.drop(['Price'], axis=1)

melb_numeric_X = melb_X.select_dtypes(exclude=['object'])

melb_numeric_X = melb_X.select_dtypes(exclude=['object'])

X_train, X_test, y_train, y_test = train_test_split(melb_numeric_X, 

                                                    y,

                                                    train_size=0.7, 

                                                    test_size=0.3, 

                                                    random_state=0)

# Function to compute the quality of prediction (MAE)

def score_dataset(X_train, X_test, y_train, y_test):

    model = RandomForestRegressor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return mean_absolute_error(y_test, preds)



cols_with_missing = [col for col in X_train.columns 

                                 if X_train[col].isnull().any()]

X_train_reduced = X_train.drop(cols_with_missing, axis=1)

X_test_reduced = X_test.drop(cols_with_missing, axis=1)



print("Dropped missing columns MAE")

MAE_DROP = (score_dataset(X_train_reduced, X_test_reduced, y_train, y_test))

print(SCORE_FROM_DROPPING)
from sklearn.impute import SimpleImputer



my_imputer = SimpleImputer()

X_train_imputed = my_imputer.fit_transform(X_train)

X_test_imputed = my_imputer.fit_transform(X_test)



print("Mean imputed MAE")

MAE_IMPUTED = score_dataset(X_train_imputed, X_test_imputed, y_train, y_test)

print(MAE_IMPUTED)
X_train_imputed_extra = X_train.copy()

X_test_imputed_extra = X_test.copy()



cols_with_missing = (col for col in X_train.columns 

                                 if X_train[col].isnull().any())

for col in cols_with_missing:

    X_train_imputed_extra[col + '_was_missing'] = X_train_imputed_extra[col].isnull()

    X_test_imputed_extra[col + '_was_missing'] = X_test_imputed_extra[col].isnull()



#imputation

my_imputer = SimpleImputer()

X_train_imputed_extra = my_imputer.fit_transform(X_train_imputed_extra)

X_test_imputed_extra = my_imputer.fit_transform(X_test_imputed_extra)



#getting score

print("Mean imputed+extra col MAE")

MAE_IMPUTED_EXTRA = score_dataset(X_train_imputed_extra, X_test_imputed_extra, y_train, y_test)

print(MAE_IMPUTED_EXTRA)
MELBOURNE.dtypes.sample(10)



one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,

                                                                    join='left', 

                                                                    axis=1)
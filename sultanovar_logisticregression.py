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

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
from path import Path

from typing import List



def read_data(root: str = '../input/hsemath2020flights',

              target_values: List[str] = ['dep_delayed_15min']):

    """

    It reads data and separate target values from features

    

    Args:

    root - path to hsemath2020flights folder: str

    target_values - list of columns that contain target values: List[str]

    

    Returns:

    train - train dataset: pd.DataFrame

    test - test dataset: pd.DataFrame

    target_value - list of target columns: List[str]

    features - list of feature columns: List[str]

    """

    root = Path(root)

    train, test = pd.read_csv(root / "flights_train.csv"), pd.read_csv(root / "flights_test.csv")

    target_value = target_values

    features = test.columns

    

    return train, test, target_value, features



base_train, base_test, TARGET_VALUE, FEATURES = read_data()





len(base_train), len(base_test)
# Разделим дату на составляющие

CATEGORICAL = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']



def split_date_to_dmy(df: pd.DataFrame):

    """it splits column DATE into 3 columns YEAR, MONTH, DAY 

    and then removes DATE column"""

    df['YEAR'] = df['DATE'].apply(lambda x: int(x.split('-')[2]))

    df['MONTH'] = df['DATE'].apply(lambda x: int(x.split('-')[1]))

    df['DAY'] = df['DATE'].apply(lambda x: int(x.split('-')[0]))

    df.drop(['DATE'], axis=1, inplace=True)

    return df



def split_dep_time_to_hm(df: pd.DataFrame):

    """it splits DEPARTURE_TIME into 2 columns: DEP_HOUR, DEP_MINUTE,

    and then removes DEPARTURE_TIME column"""

    df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].astype(int)

    df['DEP_HOUR'] = df['DEPARTURE_TIME'].apply(lambda x: int(("%04d" % x)[:-2]))

    df['DEP_HOUR'] = df['DEP_HOUR'].apply(lambda x: 0 if x == 24 else x)

    df['DEP_MINUTE'] = df['DEPARTURE_TIME'].apply(lambda x: int(("%04d" % x)[-2:]))

    df.drop(['DEPARTURE_TIME'], axis=1, inplace=True)

    return df



def cat_as_str(df: pd.DataFrame, categorical_columns: List[str]):

    """each categorical value will be present as <*> for convenient"""

    for col in categorical_columns:

        df[col] = df[col].apply(lambda x: "<" + str(x) + ">")

    return df



def base_preparation(df: pd.DataFrame):

    """it combines previous fucntions"""

    split_date_to_dmy(df)

    split_dep_time_to_hm(df)

    cat_as_str(df, CATEGORICAL)

    return df

    
base_preparation(base_train)

base_preparation(base_test)



base_train.head()
#посмотрим сколько пропусков в данных

base_train.isnull().sum(axis=0)
# статистическое описание данных

base_train.describe()
# соотношение классов

base_train.groupby(TARGET_VALUE).count()
# Изобразим данные каждого столбца



def plot_hist(column: str, data: pd.DataFrame, ax=None):

    if column in CATEGORICAL:

        sns.countplot(x=column, data=data, ax=ax)        

    else:

        sns.distplot(data[col], ax=ax)



        

def plot_conditional_hist(column: str, data: pd.DataFrame, figsize=(13, 7), cond=TARGET_VALUE[0]):

    if column in CATEGORICAL:

        g = sns.FacetGrid(data, col=cond)

        g = (g.map(sns.countplot, column))

        g.fig.set_size_inches(*figsize)

                    

    else:

        _, bins = np.histogram(data[column])

        g = sns.FacetGrid(data, col=cond)

        g = (g.map(sns.distplot, column, bins=bins))

        g.fig.set_size_inches(*figsize)
fig, axes = plt.subplots(3, 3, figsize=(23, 19))

fig.suptitle('Historgrams for columns', fontsize=16)



for i, col in enumerate(base_train.drop(TARGET_VALUE, axis=1).columns):

    plot_hist(col, base_train, ax=axes[int(i / 3)][int(i % 3)])



for i in range(3):

    plt.close(i + 2) # delete extra figures (seaborn troubles)





# Рассмотрим условные гистограммы (зависимость от целевой переменной)



for i, col in enumerate(base_train.drop(TARGET_VALUE, axis=1).columns):

    plot_conditional_hist(col, base_train, figsize=(21, 7))



from sklearn.model_selection import train_test_split



# Разделим base_train на train и test, чтобы иметь возможность оценить модель без submission

train, test = train_test_split(base_train, stratify=base_train[TARGET_VALUE], test_size=0.1)

print(len(test))

test.head()
def save_predictions(file_name, y_predict):

    """

    It writes predictions in right format to './file_name' file

        

    Args:

    file_name - the name of the file where the predictions will be saved: str

    y_predict - result of 'predict_proba' method, y_predict.shape = (n, 2): np.ndarray

    

    Returns:

    None

    """

    prediction = pd.DataFrame(y_predict[:, 1], columns=['dep_delayed_15min'])

    prediction.to_csv(file_name, index_label='id')
def cut_data(data: pd.DataFrame):

    """It drops unnecessary features"""

    new_data = data[['DISTANCE', 'MONTH']]

    new_data['TIME'] = data.apply(lambda x: int(x['DEP_HOUR']) * 100 + int(x['DEP_MINUTE']), axis=1)

    if TARGET_VALUE[0] in data.columns:

        new_data[TARGET_VALUE] = data[TARGET_VALUE]

    return new_data
train = cut_data(train)

test = cut_data(test)

base_test = cut_data(base_test)

base_test.head()
CATEGORICAL = ['MONTH']

FEATURES = base_test.columns
y_train = np.array(train[TARGET_VALUE].astype(int))

X_train = np.array(train[FEATURES])



y_test = np.array(test[TARGET_VALUE].astype(int))

X_test = np.array(test[FEATURES].astype(int))



X_base_test = np.array(base_test)



indexes_of_categorical_features = [num for num, col in enumerate(FEATURES) if col in CATEGORICAL]

indexes_of_categorical_features

indexes_of_numerical_features = [num for num, col in enumerate(FEATURES) if col not in CATEGORICAL]

indexes_of_numerical_features
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer



ct = ColumnTransformer(

    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), indexes_of_categorical_features),

                  ('num', StandardScaler(), indexes_of_numerical_features)],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])

    remainder='passthrough'                         # Leave the rest of the columns untouched

)



X_train_hat = ct.fit_transform(X_train).toarray()

X_test_hat = ct.transform(X_test).toarray()

X_base_test_hat = ct.transform(X_base_test).toarray()



X_train_hat.shape
from sklearn.linear_model import LogisticRegression



model = LogisticRegression()
from sklearn.model_selection import RandomizedSearchCV



hyperparameter = {'penalty': ['l2'], 'C': np.linspace(0.1, 1, 100)}

search = RandomizedSearchCV(model, hyperparameter, n_iter=9)

search.fit(X_train_hat, y_train.ravel())

l2 = search.best_params_

l2
model = LogisticRegression(**l2)

model.fit(X_train_hat, y_train.ravel())



y_test_predictions = model.predict_proba(X_test_hat)

y_test_predictions
from sklearn.metrics import roc_auc_score



# Посмотрим какой скор даст модель на нашем выделенном куске обучающей выборки

roc_auc_score(y_test.ravel(), y_test_predictions[:, 1])
y_base_test_predictions = model.predict_proba(X_base_test_hat)

save_predictions('./logreg_v0.0_tunnedl2_0.67.csv', y_base_test_predictions)
pd.read_csv('./logreg_v0.0_tunnedl2_0.67.csv').head()
# модель выдает 0.67706 скор.
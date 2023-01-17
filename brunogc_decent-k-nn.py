import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# For pretty for loops. Not actually necessary :)

from tqdm import tqdm_notebook



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier



# Use input to run in kaggle

DATA_PATH = "../input/"

#DATA_PATH = "../"
# Bibliotecas de visualização

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

train_df = pd.read_csv(DATA_PATH + "train.csv", index_col="RowNumber")

test_df = pd.read_csv(DATA_PATH + "test.csv", index_col="RowNumber")

sample = pd.read_csv(DATA_PATH + "sampleSubmission.csv", index_col="RowNumber")



# Drop useless cols

train_df.drop(columns="CustomerId", inplace=True)

test_df.drop(columns="CustomerId", inplace=True)



train_df.drop(columns="Surname", inplace=True)

test_df.drop(columns="Surname", inplace=True)
# Deal with NaN
# Primeiro o pais

train_df.fillna({"Geography": "France"}, inplace=True)

test_df.fillna({"Geography": "France"}, inplace=True)



# Agora o gênero

train_df.fillna({"Gender": "Male"}, inplace=True)

test_df.fillna({"Gender": "Male"}, inplace=True)
# Primeiro o pais

new_value = train_df["EstimatedSalary"].mode()

train_df.fillna({"EstimatedSalary": new_value.iloc[0]}, inplace=True)

test_df.fillna({"EstimatedSalary": new_value.iloc[0]}, inplace=True)



train_df.isna().sum().sum(), test_df.isna().sum().sum()
for col in train_df.columns:

    print(col, "\t\t", train_df[col].dtype)
for col in ["Gender", "Geography"]:

    # Criar o encoder

    encoder = LabelBinarizer()

    encoder.fit(train_df[col])

    

    # Substituir aquela coluna no treino e teste

    train_df[col] = encoder.transform(train_df[col])

    test_df[col] = encoder.transform(test_df[col])
from sklearn.preprocessing import StandardScaler

x_train = train_df.copy(deep=True)

y_train = x_train["Exited"]

x_train = x_train.drop(columns="Exited")



scaler = StandardScaler()

x_train.loc[:] = scaler.fit_transform(x_train)

x_test = pd.DataFrame(scaler.fit_transform(test_df), columns=test_df.columns,

                      index=test_df.index)
model = KNeighborsClassifier(n_jobs=-1)



grid = {'n_neighbors': range(3, 50, 3),

        'metric': ["euclidean", "manhattan", "chebyshev", "minkowski"],

        }



search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=8, 

                             cv=2, verbose=1, return_train_score=True)



search.fit(x_train, y_train)



# and after some hours...

df_search = pd.DataFrame(search.cv_results_)

df_search.sort_values("mean_test_score", inplace=True)
df_search.tail()
search.best_score_, search.best_params_
params = search.best_params_
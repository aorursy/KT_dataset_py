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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df.head()
df.Pclass.unique()
df.Sex.unique()
df.Ticket.nunique()
df.Embarked.unique()
df.Parch.unique()
cols = ['Fare', 'Embarked', 'Sex', 'Age']
X = df[cols]
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
ohe = OneHotEncoder()
imp = SimpleImputer()
values = {'Embarked': 'nan'}
X = X.fillna(values)
ct = make_column_transformer(
(ohe, ['Embarked', 'Sex']),
    (imp, ['Age', 'Fare']),
)
X_t = ct.fit_transform(X)
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_test.isna().sum()
x = X_t
y = df['Survived'].values
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from skopt.callbacks import CheckpointSaver
import time
from sklearn.model_selection import cross_val_score
DO_CV = True

space_sgdclassifier = [
    Real(1e-8, 1e-4, name="alpha"),
    Real(0.0, 1.0, name="l1_ratio"),
    Integer(1000, 15000, name="max_iter"),
    Categorical(
        ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"], name="loss"
    ),
    Categorical(["l2", "l1", "elasticnet"], name="penalty"),
]
hyperclassy = SGDClassifier()

# space_mlp = [
#     Categorical(["sgd", "adam"], name="solver"),
#     Categorical(["constant", "invscaling", "adaptive"], name="learning_rate"),
#     Categorical([(10,10,10), (20,), (20,10,5)], name="hidden_layer_sizes"),
#     Real(0.00001, 0.01, name="alpha")
# ]

# hyperclassy = MLPClassifier(
#     batch_size=10, early_stopping=True, n_iter_no_change=5, verbose=True, tol=1e-3
# )
checkpoint_path = "./{}-hyperopt-checkpoint.pkl".format(hyperclassy.__class__.__name__)
checkpoint_saver = CheckpointSaver(
    checkpoint_path, compress=9
)

@use_named_args(space_sgdclassifier)
def objective(**params):
    print(">>> Running objective... (Params: {})".format(params))
    t_start = time.time()

    hyperclassy.set_params(**params)

    if DO_CV:
        print("Doing cv..")
        scores = cross_val_score(hyperclassy, x, y, cv=3, scoring='f1_macro')
        accuracy = np.mean(scores)
    t_end = time.time()
    print("<<< Took {:.2f}s cv acc: {:.4f}\n".format(t_end - t_start, accuracy))

    return -accuracy
from skopt import gp_minimize

res_gp = gp_minimize(
    objective,
    space_sgdclassifier,
    n_calls=70,
    random_state=0,
)
best_params_list = res_gp.x

best_params = {
    "alpha": best_params_list[0],
    "l1_ratio": best_params_list[1],
    "max_iter": best_params_list[2],
    "loss": best_params_list[3],
    "penalty": best_params_list[4],
}

print("Parameters for crossval acc {:.4f}:\n{}".format(-res_gp.fun, best_params))
hyperclassy.set_params(**best_params)
hyperclassy.fit(x, y)
x_test = df_test[cols]
x_test = x_test.fillna(values)
x_test = ct.transform(x_test)
x_pred = hyperclassy.predict(x_test)
df_test['Survived'] = x_pred
df_test.head()
df_test[['PassengerId', 'Survived']].to_csv('gender_submission.csv')
df_test[['PassengerId', 'Survived']].shape

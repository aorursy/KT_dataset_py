import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv("../input/train.csv")



dataset.head()
dataset.Sex = dataset.Sex.replace("female", 0)

dataset.Sex = dataset.Sex.replace("male", 1)
y = dataset.Survived



features = ["Sex", "Age", "Parch", "SibSp"]

X = dataset[features]
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(X, y, random_state = 0)
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor



my_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.01)



my_pipeline = Pipeline(steps=[

    ("model", my_model)

])
my_pipeline.fit(X_tr, y_tr, model__early_stopping_rounds = 10, model__eval_set = [(X_val, y_val)], model__verbose = False)



preds = my_pipeline.predict(X_val)
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_val, preds)
dataset2 = pd.read_csv("../input/test.csv")



dataset2.Sex = dataset2.Sex.replace("female", 0)

dataset2.Sex = dataset2.Sex.replace("male", 1)



X = dataset2[features]



preds2 = my_pipeline.predict(X)
survivors  = pd.Series(preds2, name = "Survived")



survivors.head()
def is_alive(x):

    if x > 0.5:

        return 1

    else:

        return 0



survivors = survivors.map(lambda x: is_alive(x))



survivors.head()
submission = dataset2.PassengerId.to_frame().join(survivors)

submission.describe()
submission.to_csv('submission.csv', index=False)
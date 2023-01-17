import pandas as pd
from collections import defaultdict
from statistics import mean
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
train_data = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/train-data.csv", index_col=0)
goal_variable = ['G3']
numeric_features = ['studytime', 'age']
# categorical_features = [ 'school', 'sex', 'address', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian','failures', 'schoolsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic' ]
categorical_features = ['failures']

train_data = train_data.drop(columns=['reason', 'absences', 'famsize', 'famsup', 'famrel'])
train_data = train_data[goal_variable + numeric_features + categorical_features]
X_train_data = train_data[numeric_features + categorical_features].dropna()
y_train_data = train_data[goal_variable].to_numpy()


X_train, X_dev, y_train, y_dev = train_test_split(X_train_data, y_train_data, test_size=0.1)

pipeline = Pipeline([
    ('pre', make_column_transformer((OneHotEncoder(handle_unknown='ignore'), categorical_features), remainder='passthrough')),
    ('clf', LinearRegression())
])

pipeline.fit(X_train, y_train)
y_train_pred = pipeline.predict(X_train)
print("MAE auf dem Training Datensatz\t\t", mean_absolute_error(y_train, y_train_pred))

y_dev_pred = pipeline.predict(X_dev)
print("MAE auf dem Validation Datensatz\t", mean_absolute_error(y_dev, y_dev_pred))
X_test = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/test-data.csv", index_col=0)
X_test = X_test[numeric_features + categorical_features]
y_test_pred = pipeline.predict(X_test)
X_test_submission = pd.DataFrame(index=X_test.index)
X_test_submission['G3'] = y_test_pred
X_test_submission.to_csv('linear_submission.csv', header=True, index_label='id')
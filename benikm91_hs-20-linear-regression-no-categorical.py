import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
train_data = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/train-data.csv")
# reason not used as to many values missing.
train_data = train_data.drop(columns=['reason'])

X_train = train_data.drop(columns=['G1', 'G2', 'G3'])[
    ['age', 'studytime', 'traveltime', 'absences']
]
y_train = train_data['G3'].to_numpy()
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
pipeline = Pipeline([
    ('clf', LinearRegression())
])
pipeline.fit(X_train, y_train)
y_dev_pred = pipeline.predict(X_dev)
print('MAE\t', mean_absolute_error(y_dev, y_dev_pred))
X_test = pd.read_csv("/kaggle/input/machine-learning-lab-cas-data-science-hs-20/test-data.csv", index_col=0)[['age', 'studytime', 'traveltime', 'absences']]
X_test.describe()
y_test_pred = pipeline.predict(X_test)
X_test_submission = pd.DataFrame(index=X_test.index)
X_test_submission['G3'] = y_test_pred
X_test_submission.to_csv('linear_submission.csv', header=True, index_label='id')
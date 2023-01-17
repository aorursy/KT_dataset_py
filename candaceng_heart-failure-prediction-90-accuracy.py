import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
df = pd.read_csv('/kaggle/input/heart-disease/heart.csv')
df.head()
df.info()
sns.heatmap(df.corr())
X = df.loc[:, df.columns != 'target']
y = df.loc[:, 'target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model1 = RandomForestRegressor()
bags = 10
seed = 1
bagged_prediction = np.zeros(X_val.shape[0])
for n in range(0, bags):
    model1.set_params(random_state = seed + n) 
    model1.fit(X_train, y_train)
    preds1 = model1.predict(X_val)
    bagged_prediction += preds1
bagged_prediction /= bags
test_preds1 = model1.predict(X_test)
df['log_cp'] = np.log(df['cp'] + 1)
df['log_ca'] = np.log(df['ca'] + 1)
df['log_oldpeak'] = np.log(df['oldpeak'] + 1)
df['log_trestbps'] = np.log(df['trestbps'] + 1)
df['log_thalach'] = np.log(df['thalach'] + 1)
df.drop('cp', axis=1, inplace=True)
df.drop('ca', axis=1, inplace=True)
df.drop('oldpeak', axis=1, inplace=True)
df.drop('trestbps', axis=1, inplace=True)
df.drop('thalach', axis=1, inplace=True)
X = df.loc[:, df.columns != 'target']
y = df.loc[:, 'target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model2 = LinearRegression().fit(X_train, y_train)
preds2 = model2.predict(X_val)
test_preds2 = model2.predict(X_test)

stacked_predictions = np.column_stack((preds1, preds2))
stacked_test_predictions = np.column_stack((test_preds1, test_preds2))
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_val)
final_predictions = meta_model.predict(stacked_test_predictions)
final_predictions = (final_predictions >= 0.5).astype(int)
final_predictions
f1_score(final_predictions, y_test)
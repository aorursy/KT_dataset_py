import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
train = df.loc[:399, :]
val = df.loc[400:, :]
print(train.info())
print(val.info())
train.head()
train.drop('Serial No.', axis=1, inplace=True)
val.drop('Serial No.', axis=1, inplace=True)
sns.heatmap(train.corr())
train.drop('Research', axis=1, inplace=True)
val.drop('Research', axis=1, inplace=True)
sns.pairplot(train)
X_test = val.loc[:, val.columns != 'Chance of Admit ']
y_test = val.loc[:, 'Chance of Admit ']
X = train.loc[:, train.columns != 'Chance of Admit ']
y = train.loc[:, 'Chance of Admit ']
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
train['log_GRE'] = np.log(train['GRE Score'] + 1)
train['log_TOEFL'] = np.log(train['TOEFL Score'] + 1)
train['log_CGPA'] = np.log(train['CGPA'] + 1)
train.drop('GRE Score', axis=1, inplace=True)
train.drop('TOEFL Score', axis=1, inplace=True)
train.drop('CGPA', axis=1, inplace=True)

val['log_GRE'] = np.log(val['GRE Score'] + 1)
val['log_TOEFL'] = np.log(val['TOEFL Score'] + 1)
val['log_CGPA'] = np.log(val['CGPA'] + 1)
val.drop('GRE Score', axis=1, inplace=True)
val.drop('TOEFL Score', axis=1, inplace=True)
val.drop('CGPA', axis=1, inplace=True)
X_test = val.loc[:, val.columns != 'Chance of Admit ']
y_test = val.loc[:, 'Chance of Admit ']
X = train.loc[:, train.columns != 'Chance of Admit ']
y = train.loc[:, 'Chance of Admit ']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model2 = LinearRegression().fit(X_train, y_train)
preds2 = model2.predict(X_val)
test_preds2 = model2.predict(X_test)

stacked_predictions = np.column_stack((preds1, preds2))
stacked_test_predictions = np.column_stack((test_preds1, test_preds2))
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_val)
final_predictions = meta_model.predict(stacked_test_predictions)
final_predictions
mean_squared_error(final_predictions, y_test, squared=False)
import numpy as np

import pandas as pd 

from sklearn import linear_model

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



df_train =pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# df_train.head()
# We need only continous columns

contCols = df_train.columns[df_train.columns.str.startswith('cont')]
x_train = df_train[contCols]

y_train = df_train['loss']



x_test = df_test[contCols]

id_test = df_test['id']
# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(x_train, y_train)





# Predict using the trained model 

y_pred = regr.predict(x_train)
diff = y_pred - y_train

diff.describe()



type(y_pred)
print(mean_absolute_error(y_train, y_pred))

print(mean_squared_error(y_train, y_pred))

print(r2_score(y_train, y_pred))
df_test.columns
# TODO find performance

# TODO plot data and results

# TODO how to evaluate the performance?
whole = df_train[df_train.columns[df_train.columns.str.startswith('cont')]].copy()

whole['loss'] = df_train.loss

whole.head()
corr = whole.corr()
corr['loss'].sort_values(ascending= False)

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corr)
x_train.corrwith(y_train).sort_values(ascending= False)
x_train.describe()
y_train.describe()
import pandas as pd

import numpy as np

from sklearn import preprocessing, model_selection

import sklearn

from catboost import CatBoostRegressor

from sklearn.metrics import mean_absolute_error

import os

import matplotlib.pyplot as plt
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/new_data_99_06_03_13_04.csv', delimiter=',')

data.shape

data.head(10)
len(data) - data.count()
df = data[:].nunique()

df
data = data.drop(['Таможня', 'description'], axis='columns', inplace=False)

data = data.dropna()
len(data) - data.count()

data.dtypes
le = preprocessing.LabelEncoder()

categorical_columns = data.columns[data.dtypes == 'object']



for column in categorical_columns:

    data[column] = le.fit_transform(list(data[column]))
data.dtypes
fig = plt.figure(figsize=(15,8))

ax1 = fig.add_subplot(111)

plt.imshow(data.corr(), cmap='hot', interpolation='nearest')

plt.colorbar()

labels = data.columns

ax1.set_xticks(np.arange(len(labels)))

ax1.set_yticks(np.arange(len(labels)))

ax1.set_xticklabels(labels,rotation=90, fontsize=10)

ax1.set_yticklabels(labels,fontsize=10)

plt.show()
predict = 'Price'



X = np.array(data.drop([predict], 1))

y = np.array(data[predict])



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)



model = CatBoostRegressor(learning_rate=0.5)

model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)

print('Accuracy of model:', accuracy)



predictions = model.predict(x_test)

mae = mean_absolute_error(predictions, y_test)

print("Mean Absolute Error:", mae)
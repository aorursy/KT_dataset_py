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
import numpy as np

import seaborn as sns

from sklearn.model_selection import KFold, train_test_split

from sklearn.linear_model import SGDClassifier, LinearRegression

from sklearn.metrics import log_loss, accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error, mean_absolute_error
df_data = pd.read_csv('../input/kickstarter-projects/ks-projects-201801.csv')

df_data.head(30)
df_data.describe()
#convert qualitative variable to dummy variable

df_data = pd.get_dummies(df_data, columns = ["category", "main_category", "currency", "country", "state"])

df_data["pledged_per_goal"] = df_data["usd_pledged_real"] / df_data["usd_goal_real"]

df_data.head()
sns.heatmap(pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('category')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr())



pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('category')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr()
sns.heatmap(pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('main_category')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr())



pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('main_category')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr()
sns.heatmap(pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('currency')],

    df_data.loc[:, df_data.columns.str.startswith('country')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"]

], axis = 1).corr())



pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('currency')],

    df_data.loc[:, df_data.columns.str.startswith('country')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"]

], axis = 1).corr()
sns.heatmap(pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('usd_goal_real')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr())



pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('usd_goal_real')],

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr()
sns.heatmap(pd.concat([

    pd.to_datetime(df_data["launched"]).map(pd.Timestamp.timestamp),

    pd.to_datetime(df_data["deadline"]).map(pd.Timestamp.timestamp),

    pd.to_datetime(df_data["deadline"]).map(pd.Timestamp.timestamp) - pd.to_datetime(df_data["launched"]).map(pd.Timestamp.timestamp),

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr())



pd.concat([

    pd.to_datetime(df_data["launched"]).map(pd.Timestamp.timestamp),

    pd.to_datetime(df_data["deadline"]).map(pd.Timestamp.timestamp),

    pd.to_datetime(df_data["deadline"]).map(pd.Timestamp.timestamp) - pd.to_datetime(df_data["launched"]).map(pd.Timestamp.timestamp),

    df_data.loc[:, df_data.columns.str.startswith('state')],

    df_data["usd_pledged_real"],

    df_data["pledged_per_goal"]

], axis = 1).corr()
df_input = pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('category_')],

    df_data.loc[:, df_data.columns.str.startswith('main_category_')],

    df_data.loc[:, df_data.columns.str.startswith('country_')],

    df_data.loc[:, df_data.columns.str.startswith('currency_')],

    df_data.loc[:, df_data.columns.str.startswith('usd_goal_real_')],

    pd.to_datetime(df_data["launched"]).map(pd.Timestamp.timestamp),

    pd.to_datetime(df_data["deadline"]).map(pd.Timestamp.timestamp),

    pd.to_datetime(df_data["deadline"]).map(pd.Timestamp.timestamp) - pd.to_datetime(df_data["launched"]).map(pd.Timestamp.timestamp)

], axis = 1)

df_input.head()
df_target = pd.concat([

    df_data.loc[:, df_data.columns.str.startswith('state_')],

    df_data[["usd_pledged_real", "pledged_per_goal", "backers"]],

], axis = 1)

df_target.head(30)
df_target.describe()
df_target1 = df_data.loc[:, df_data.columns.str.startswith('state_')]
na_data_x = df_input.values

na_data_y1 = np.argmax(df_target1.values, axis = 1)

print(na_data_y1)
X_train, X_test, Y_train, Y_test = train_test_split(na_data_x, na_data_y1, test_size = 0.1)
clf = SGDClassifier(loss = 'log', penalty = 'none', max_iter = 10000, fit_intercept = True, random_state = 1234, tol = 1e-3)

clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

precision, recall, f1_score, _ = precision_recall_fscore_support(Y_test, y_pred)
print('Accuracy: {:.3f}%'.format(100 * accuracy))

print('Precision: {:.3f}%'.format(100 * precision[0]))

print('Recall: {:.3f}%'.format(100 * recall[0]))

print('F1 Score: {:.3f}%'.format(100 * f1_score[0]))
conf_mat = confusion_matrix(Y_test, y_pred)

conf_mat = pd.DataFrame(conf_mat,

                           index = ['正解: canceled', '正解: failed', '正解: live', '正解: successful', '正解: suspended', '正解: undefined'],

                           columns = ['予測: canceled', '予測: failed', '予測: live', '予測: successful', '予測: suspended', '予測: undefined']

)

conf_mat
df_target2 = df_target["usd_pledged_real"]
na_data_x = df_input.values

na_data_y2 = df_target2.values
X_train, X_test, Y_train, Y_test = train_test_split(na_data_x, na_data_y2, test_size = 0.1)
regr = LinearRegression(fit_intercept = True)

regr.fit(X_train, Y_train)
y_pred = regr.predict(X_test)



mse = mean_squared_error(Y_test, y_pred)

mae = mean_absolute_error(Y_test, y_pred)

rmse = np.sqrt(mse)
print("MSE: %s"%round(mse, 3))

print("MAE: %s"%round(mae, 3))

print("RMSE: %s"%round(rmse, 3))
df_target3 = df_target["pledged_per_goal"]
na_data_x = df_input.values

na_data_y3 = df_target3.values
X_train, X_test, Y_train, Y_test = train_test_split(na_data_x, na_data_y3, test_size = 0.1)
regr = LinearRegression(fit_intercept = True)

regr.fit(X_train, Y_train)
df_input.head(10)
y_pred = regr.predict(X_test)



mse = mean_squared_error(Y_test, y_pred)

mae = mean_absolute_error(Y_test, y_pred)

rmse = np.sqrt(mse)
print("MSE: %s"%round(mse, 3))

print("MAE: %s"%round(mae, 3))

print("RMSE: %s"%round(rmse, 3))
print("MAE of real value: %s"%round(mean_absolute_error(Y_test, y_pred * X_test[:, -4]), 3))
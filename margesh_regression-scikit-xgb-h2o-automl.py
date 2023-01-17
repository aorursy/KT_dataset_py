import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

data.head()
data.shape
data.isna().sum()
data = data.drop('Subtitle', axis=1)

filtered_data = data.dropna()

filtered_data.shape
filtered_data.dtypes == 'object'
filtered_data['Genres'].value_counts()
genres = filtered_data['Genres'].str.split(', ')

genres
from sklearn.preprocessing import MultiLabelBinarizer



# Binarise labels



mlb = MultiLabelBinarizer()

expandedLabelData = mlb.fit_transform(filtered_data['Genres'].str.split(', '))

labelClasses = mlb.classes_

print(labelClasses)



# Create a pandas.DataFrame from our output

expandedLabels = pd.DataFrame(expandedLabelData, columns=labelClasses)

expandedLabels.head()
expandedLabels.sum()
categories = list(expandedLabels.columns.values)

plt.figure(figsize=(20,10))

f = sns.barplot(categories, expandedLabels.sum().values)

f.set_xticklabels(f.get_xticklabels(), rotation='25', ha="right");
expanded_labels_minus_games_strategy = expandedLabels.drop(["Strategy", "Games"], axis=1)

categories = list(expanded_labels_minus_games_strategy.columns.values)

plt.figure(figsize=(20,10))

f = sns.barplot(categories, expanded_labels_minus_games_strategy.sum().values)

f.set_xticklabels(f.get_xticklabels(), rotation='25', ha="right");
print(filtered_data.shape)

print(expandedLabels.shape)

df = pd.concat([filtered_data.reset_index(drop=True),expandedLabels.reset_index(drop=True)], axis=1)

print(df.shape)

df.head()
df = df.drop(['URL', 'ID', 'Icon URL'], axis=1)

df.head()
df['In-app Purchases'].value_counts()
mlb_inapp = MultiLabelBinarizer()

expandedLabelData = mlb_inapp.fit_transform(df['In-app Purchases'].str.split(', '))

labelClasses = mlb_inapp.classes_

print(labelClasses)



# Create a pandas.DataFrame from our output

expandedLabels = pd.DataFrame(expandedLabelData, columns=labelClasses)

expandedLabels.head()
max_10_prices = expandedLabels.sum().sort_values(ascending=False)[:10]

sns.barplot(max_10_prices.index, max_10_prices.values)
df = pd.concat([df.reset_index(drop=True),expandedLabels.reset_index(drop=True)], axis=1)
y = df['Average User Rating']
desc = df['Description']

desc_lengths = [len(de) for de in desc]

df['desc_lengths'] = desc_lengths
df = df.drop(['Name', 'Average User Rating', 'In-app Purchases', 'Languages', 'Genres', 'Description'], axis=1)

df.head()
from datetime import datetime

date_format = "%d/%m/%Y"

(datetime.strptime(df['Current Version Release Date'][1], date_format) - datetime.strptime(df['Original Release Date'][1], date_format)).days
dataset_date = datetime.strptime("3/7/2019", date_format)

dataset_date
curr_minus_orig = []



for i in range(len(df)):

    curr = df['Current Version Release Date'][i]

    orig = df['Original Release Date'][i]

    diff = datetime.strptime(curr, date_format) - datetime.strptime(orig, date_format)

    curr_minus_orig.append(diff.days)

    



df['Current minus Original'] = np.array(curr_minus_orig)

df['Original Release Date'] = np.array([ (dataset_date - datetime.strptime(date, date_format)).days for date in df['Original Release Date']])

df['Current Version Release Date'] =  np.array([ (dataset_date - datetime.strptime(date, date_format)).days for date in df['Current Version Release Date']])
df.dtypes[df.dtypes == object]
df['Age Rating'].unique()
from sklearn import preprocessing

# Label Encoding

for f in df.columns:

    if df[f].dtype=='object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[f].values))

        df[f] = lbl.transform(list(df[f].values))
df['Age Rating'].unique()
df['Age Rating'][df['Age Rating'] == 2] = 4

df['Age Rating'][df['Age Rating'] == 3] = 9

df['Age Rating'][df['Age Rating'] == 0] = 12

df['Age Rating'][df['Age Rating'] == 1] = 17
df['Age Rating'].unique()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=123)
X_train = X_train.fillna(0)

X_test = X_test.fillna(0)
X_train.shape
sns.distplot(y, hist=True, kde=False, bins=9, hist_kws={'edgecolor':'#000000'})
X_train.head()
X_test.head()
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



X_train_preprocessed = preprocessing.normalize(X_train)

X_test_preprocessed = preprocessing.normalize(X_test)



lin_model = LinearRegression()

lin_model.fit(X_train_preprocessed, y_train)



y_train_predict = lin_model.predict(X_train_preprocessed)



rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))



print("The model performance for training set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))

print("\n")



# model evaluation for testing set

y_test_predict = lin_model.predict(X_test_preprocessed)

rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))



print("The model performance for testing set")

print("--------------------------------------")

print('RMSE is {}'.format(rmse))
error_frame = pd.DataFrame({'Actual': np.array(y_test).flatten(), 'Predicted': y_test_predict.flatten()})

error_frame.head(10)
df1 = error_frame[0:20]

df1.plot(kind='bar',figsize=(24,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
import xgboost as xgb



xgr = xgb.XGBRegressor(           

                 learning_rate=0.05,

                 max_depth=8,

                 min_child_weight=1.5,

                 n_estimators=500,                                                                    

                 seed=42,

                 num_boost_rounds=50,

                 objective="reg:squarederror",

                 tree_method='gpu_hist',  #IMPORTANT. GPU_HIST NEEDS GPU, OTHERWISE ERROR WILL BE THROWN.

                )

xgr.fit(X_train, y_train)
train_pred = xgr.predict(data= X_train)

test_pred = xgr.predict(data= X_test)



mse_train = mean_squared_error(y_train, train_pred)

mse_test = mean_squared_error(y_test, test_pred)



print('RMSE train : {:.3f}'.format(np.sqrt(mse_train)))

print('RMSE test : {:.3f}'.format(np.sqrt(mse_test)))
from sklearn.model_selection import GridSearchCV



params = {'learning_rate': [0.01, 0.03, 0.06],

          'max_depth' : [4, 6, 8],

          'n_estimators' : [250, 500, 1000, 1500, 2000],

          'num_boost_rounds' : [5, 20],}
%%time

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



xgr_optimized = xgb.XGBRegressor(objective="reg:squarederror",

                                 min_child_weight=1.5,

                                 tree_method='gpu_hist')



grid = GridSearchCV(estimator=xgr_optimized, scoring="neg_mean_squared_error", param_grid = params, verbose=1, cv=3)

grid.fit(X_train, y_train)
print(grid.best_params_)
xgr_best = xgb.XGBRegressor(learning_rate = 0.01, max_depth  = 6, n_estimators =  500, num_boost_rounds =  5,  min_child_weight=1.5, tree_method='gpu_hist') #, tree_method='gpu_hist') # You canuncomment this tree method part if you are using GPU



xgr_best.fit(X_train, y_train)
train_pred = xgr_best.predict(data= X_train)

test_pred = xgr_best.predict(data= X_test)



mse_train = mean_squared_error(y_train, train_pred)

mse_test = mean_squared_error(y_test, test_pred)



print('RMSE train : {:.3f}'.format(np.sqrt(mse_train)))

print('RMSE test : {:.3f}'.format(np.sqrt(mse_test)))
error_frame = pd.DataFrame({'Actual': np.array(y_test).flatten(), 'Predicted': test_pred.flatten()})

error_frame.head(10)
df1 = error_frame[:20]

df1.plot(kind='bar',figsize=(24,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
ax = xgb.plot_importance(xgr_best, max_num_features=10)

plt.figure(figsize=(15,25))

plt.show()
# ax = xgb.plot_importance(xgr, max_num_features=10)

# plt.figure(figsize=(15,25))

# plt.show()
import h2o

from h2o.automl import H2OAutoML

h2o.init(max_mem_size='11G')
X_train['target'] = y_train

h2o_train = h2o.H2OFrame(X_train)



X_test['target'] = y_test

h2o_test = h2o.H2OFrame(X_test)
h2o_train.head()
aml = H2OAutoML(max_runtime_secs=1200, seed=1)

aml.train(x=list(X_train.columns), y="target", training_frame=h2o_train, validation_frame=h2o_test)
lb = aml.leaderboard

lb.head()
aml.predict(h2o_test)
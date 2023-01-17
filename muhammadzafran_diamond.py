# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
diamond = pd.read_csv('../input/diamonds.csv')

diamond.head()
from sklearn.model_selection import train_test_split
index = diamond.index

columuns = diamond.columns

data = diamond.values
diamond = diamond.drop('Unnamed: 0', axis='columns')
diamond.head()
index = diamond.index

columuns = diamond.columns

data = diamond.values
diamond_num = diamond.select_dtypes(include=[np.number])
train_set, test_set = train_test_split(diamond, test_size=0.2, random_state=42)
import matplotlib.pyplot as plt
def plot_histogram(xlabel, title, values, n_bins):

    plt.hist(values, bins=n_bins)

    plt.xlabel(xlabel)

    plt.ylabel("Frequency")

    plt.title(title)

    plt.axvline(x=values.mean(), linewidth=2, color='r', label="mean")

    plt.axvline(x=values.median(), linewidth=2, color='y',label="median")

    for mode in values.mode():

        plt.axvline(x=mode, linewidth=2, color='k',label="mode")

    plt.legend()

diamond_num.head()
print("Minimum :", diamond.carat.min())

print("Maximum :", diamond.carat.max())

print("Range : ", diamond.carat.max() - diamond.carat.min())

plot_histogram(diamond.carat.name,"Histogram of "+diamond.carat.name, diamond_num[diamond.carat.name], 20)
print("Minimum :", diamond.depth.min())

print("Maximum :", diamond.depth.max())

print("Range : ", diamond.depth.max() - diamond.depth.min())

plot_histogram(diamond.depth.name,"Histogram of "+diamond.depth.name, diamond_num[diamond.depth.name], 20)
print("Minimum :", diamond.table.min())

print("Maximum :", diamond.table.max())

print("Range : ", diamond.table.max() - diamond.table.min())

plot_histogram(diamond.table.name,"Histogram of "+diamond.table.name, diamond_num[diamond.table.name], 20)
print("Minimum :", diamond.price.min())

print("Maximum :", diamond.price.max())

print("Range : ", diamond.price.max() - diamond.price.min())

plot_histogram(diamond.price.name,"Histogram of "+diamond.price.name, diamond_num[diamond.price.name], 200)
import scipy as sp
diamond_num_z_scores = sp.stats.zscore(diamond_num)
plt.boxplot(diamond_num_z_scores)

plt.xticks(np.arange(1,8),diamond_num.columns)

plt.xlabel("Variables")

plt.ylabel("z-scores")

plt.title("Box Plot for Continuous Variables")
diamond_quali = diamond.select_dtypes(include=[np.object])
diamond_quali.head()
diamond_quali.cut.value_counts().plot(kind="bar")
diamond.color.value_counts().plot(kind='bar')
diamond.clarity.value_counts().plot(kind='bar')
corr = diamond_num.corr()
plt.figure(figsize=(12,9))

plt.imshow(corr, cmap='hot')

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns, rotation=20)

plt.yticks(range(len(corr)), corr.columns)

plt.title("Correlation Matrix for Continuous Variables")
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelBinarizer

mean = 0

std = 0



class DataframeTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,test=False, label_column='price'):

        self.test = test

        self.label_column = label_column

        

    def seperate_quantitative_and_qualitative(self):

        quantitative = self.dataframe.select_dtypes(include=[np.number])

        qualitative = self.dataframe.select_dtypes(include=[np.object])

        return quantitative, qualitative

    

    def normalize_quantitative_values(self, data):

        ##if it is for training, calculate the mean and standard deviation and save it.

        if not self.test:

            global mean

            mean = data.mean()

            global std

            std = data.std()

            scaler = StandardScaler()

            data_normalize = scaler.fit_transform(data)

        elif self.test: #if test, use the same mean and standard deviation obtained from training data

            if mean.empty or std.empty:

                raise Exception('Training data had not been created')

            else:

                data_normalize = (data-mean)/std    

        return data_normalize

    

    def encode_text_attributes(self,data):

        cat = np.empty((data.shape[0],0))

        encoder = LabelBinarizer()

        for column in data.columns:

            cat = np.c_[cat, encoder.fit_transform(data[column])]

        return cat

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        

        X = X.dropna()

        ##get the label

        self.label = X[self.label_column]

        self.dataframe = X.drop(self.label_column, axis='columns')     

        

        ##seperate the quantitative values and  qualitative values

        quantitative_data, qualitative_data = self.seperate_quantitative_and_qualitative()

        normalized_data = self.normalize_quantitative_values(quantitative_data)

        encoded_text_data = self.encode_text_attributes(qualitative_data)

        prepared_data = np.c_[normalized_data, encoded_text_data]

        return prepared_data, self.label    
datatransformer = DataframeTransformer()

training_prepared, label = datatransformer.fit_transform(train_set)
training_prepared.shape
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

lin_reg.fit(training_prepared, label)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg, training_prepared, label, 

                         scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-scores)
def display_scores(scores):

    print("Scores : ",scores)

    print("Mean : ", scores.mean())

    print("Standard Deviation : ", scores.std())
display_scores(lin_rmse_scores)
from sklearn.preprocessing import PolynomialFeatures
poly_2_deg = PolynomialFeatures(degree=2, include_bias=False) 

full_diamond_features_poly_deg_2 = poly_2_deg.fit_transform(training_prepared)
polynomial_2_deg_model = lin_reg.fit(training_prepared, label)
scores = cross_val_score(polynomial_2_deg_model, training_prepared, label, 

                         scoring="neg_mean_squared_error", cv=5)

deg_two_poly_rmse_scores = np.sqrt(-scores)
display_scores(deg_two_poly_rmse_scores)
from sklearn.model_selection import GridSearchCV
param_grid = [{

    'alpha':[0.1, 0.01, 0.001, 0.0001, 0.000001, 0]

}]
from sklearn.linear_model import Ridge
ridge = Ridge()
grid_search_ridge = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_ridge.fit(training_prepared, label)
grid_search_ridge.best_params_
ridge_model = Ridge(alpha=1e-06)
ridge_model.fit(training_prepared, label)
scores = cross_val_score(ridge_model, training_prepared, label, 

                         scoring="neg_mean_squared_error", cv=10)

ridge_model_rmse_scores = np.sqrt(-scores)
display_scores(ridge_model_rmse_scores)
from sklearn.linear_model import Lasso
param_grid = [{

    'alpha':[1, 2, 5, 10, 12]

}]
lasso_reg = Lasso()

grid_search_lasso = GridSearchCV(lasso_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search_lasso.fit(training_prepared, label)
grid_search_lasso.best_params_
lasso_model = grid_search_lasso.estimator
scores = cross_val_score(lasso_model, training_prepared, label, 

                         scoring="neg_mean_squared_error", cv=10)

lasso_model_rmse_scores = np.sqrt(-scores)
display_scores(lasso_model_rmse_scores)
from tensorflow.keras import models

from tensorflow.keras import layers
model = models.Sequential()

model.add(layers.Dense(26, activation='relu', input_shape=(26,)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(16, activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(8, activation='relu'))

model.add(layers.Dense(1, activation='relu'))
from tensorflow.keras import optimizers, losses, metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.01),

             loss=losses.MSE,

             metrics=[metrics.MSE])
X_val = training_prepared[:10000]

y_val = label[:10000]

X_train = training_prepared[10000:]

y_train = label[10000:]
history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_val, y_val))
history_dict = history.history

loss_values = np.sqrt(history_dict['loss'])

val_loss_values = np.sqrt(history_dict['val_loss'])
epochs = range(1, len(history_dict['loss'])+1)
plt.plot(epochs, loss_values, 'bo', label='Training Loss')

plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()
display_scores(val_loss_values)
def nn_k_fold_cv(model, n_folds=4):

    k = 4

    num_val_samples = len(training_prepared) // k

    num_epochs = 100

    all_scores = []

    for i in range(n_folds):

        print("Processing Fold # ", i)

        val_data = training_prepared[i * num_val_samples: (i + 1) * num_val_samples]

        val_target = label[i * num_val_samples: (i + 1) * num_val_samples]

    

        partial_train_data = np.concatenate(

            [training_prepared[: i * num_val_samples],

            training_prepared[(i + 1) * num_val_samples:]

        ], 

        axis=0)

    

        partial_train_targets = np.concatenate([

            label[:i * num_val_samples],

            label[(i + 1) * num_val_samples:]

        ], axis=0)

    

        model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=512, verbose=0)

        val_mse, val_mae = model.evaluate(val_data, val_target, verbose=0) 

        all_scores.append(val_mse)

    return all_scores

scores = nn_k_fold_cv(model)
display_scores(np.sqrt(scores))
model = models.Sequential()

model.add(layers.Dense(26, activation='relu', input_shape=(26,)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(8, activation='relu'))

model.add(layers.Dense(1, activation='relu'))
model.compile(optimizer=optimizers.RMSprop(lr=0.01),

             loss=losses.MSE,

             metrics=[metrics.MSE])
scores = nn_k_fold_cv(model)
display_scores(np.sqrt(scores))
model = models.Sequential()

model.add(layers.Dense(26, activation='relu', input_shape=(26,)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(4, activation='relu'))

model.add(layers.Dense(1, activation='relu'))
model.compile(optimizer=optimizers.RMSprop(lr=0.01),

             loss=losses.MSE,

             metrics=[metrics.MSE])
scores = nn_k_fold_cv(model)
display_scores(np.sqrt(scores))
model = models.Sequential()

model.add(layers.Dense(26, activation='relu', input_shape=(26,)))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(2, activation='relu'))

model.add(layers.Dense(1, activation='relu'))
model.compile(optimizer=optimizers.RMSprop(lr=0.01),

             loss=losses.MSE,

             metrics=[metrics.MSE])
scores = nn_k_fold_cv(model)
display_scores(np.sqrt(scores))
testdatatransformer = DataframeTransformer(test=True)
test_prepared, test_label = testdatatransformer.fit_transform(test_set)
prediction = model.predict(test_prepared)
from sklearn.metrics import mean_squared_error
nn_mse = mean_squared_error(prediction, test_label)
print("Test RMSE", np.sqrt(nn_mse))
import xgboost as xgb
def xgb_k_fold_cv(xgb_model, n_folds=4):

    k = 4

    num_val_samples = len(training_prepared) // k

    all_scores = []

    for i in range(n_folds):

        print("Processing Fold # ", i)

        val_data = training_prepared[i * num_val_samples: (i + 1) * num_val_samples]

        val_target = label[i * num_val_samples: (i + 1) * num_val_samples]

    

        partial_train_data = np.concatenate(

            [training_prepared[: i * num_val_samples],

            training_prepared[(i + 1) * num_val_samples:]

        ], 

        axis=0)

    

        partial_train_targets = np.concatenate([

            label[:i * num_val_samples],

            label[(i + 1) * num_val_samples:]

        ], axis=0)

        

        xgb_model.fit(partial_train_data, partial_train_targets)

        predict = xgb_model.predict(val_data)

        val_mse = mean_squared_error(predict, val_target)

        all_scores.append(val_mse)

    return all_scores
diamond_xgb = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.6, 

                               learning_rate=0.1, max_depth=12, alpha=5,

                              n_estimators=100)
scores = xgb_k_fold_cv(diamond_xgb)
display_scores(np.sqrt(scores))
diamond_xgb.fit(training_prepared, label)
test_predict = diamond_xgb.predict(test_prepared)
print("XGBoost Test RMSE", np.sqrt(mean_squared_error(test_predict, test_label)))
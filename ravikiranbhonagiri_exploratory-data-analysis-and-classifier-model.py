import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
! pip install pandas --upgrade
import pandas as pd

import numpy as np

import seaborn as sns #visualisation

import matplotlib.pyplot as plt #visualisation



import math, time, random, datetime

import seaborn as sns

import missingno



from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize





from sklearn.model_selection import train_test_split

from sklearn import model_selection, tree, preprocessing, metrics, linear_model

from sklearn.svm import LinearSVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier

from sklearn.tree import DecisionTreeClassifier



import warnings

warnings.filterwarnings('ignore')

%load_ext google.colab.data_table

plt.style.use('seaborn-whitegrid')

%matplotlib inline 

sns.set(color_codes=True)
df_train = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")

df_test = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")

df_sub = pd.read_csv("/kaggle/input/forest-cover-type-prediction/sampleSubmission.csv")
df_train.head(10)
df_test.head()
df_sub.head()
df_train.dtypes
df_train.isna().sum()
df_train.columns
df_train.describe()
con_columns = [ 'Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']

    

cat_columns = [ 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']



out_label = 'Cover_Type'
df_train[con_columns].describe()
df_train.head()
def plot_continous_data( df, columnlist, nbins=50):

  count = 0

  for col in columnlist:

    for index in range(2):

      plt.figure(count, figsize=(10,5))

      count += 1

      #sns.distplot(df_train[col])

      if index == 0:

        df_train[col].plot.hist(bins=nbins)

      else:

        sns.distplot(df_train[col])

plot_continous_data(df_train, con_columns, 50)
def box_plot_continous_data( df, columnlist):

  count = 0

  for col in columnlist:

    for index in range(2):

      plt.figure(count, figsize=(10,5))

      count += 1

      sns.boxplot(df_train[col])
box_plot_continous_data(df_train, con_columns)
for index in range(len(con_columns)):

  plt.figure(index, figsize=(10,5))

  sns.boxplot(y = df_train[con_columns[index]], x = df_train[out_label])
for index in range(len(con_columns)):

  plt.figure(index, figsize=(10,5))

  sns.violinplot(y = df_train[con_columns[index]], x = df_train[out_label])
con_columns
collist = con_columns[:3]

collist.append(out_label)

sns.pairplot(df_train[ collist ], hue = "Cover_Type", diag_kind="hist")

collist = con_columns[3:6]

collist.append(out_label)

sns.pairplot(df_train[ collist ], hue = "Cover_Type", diag_kind="hist")
collist = con_columns[6:10]

collist.append(out_label)

sns.pairplot(df_train[ collist ], hue = "Cover_Type", diag_kind="hist")
for index in range(len(cat_columns)):

  plt.figure(index, figsize=(10,5))

  #print(df_train.groupby(out_label)[cat_columns[index]].value_counts())

  #print(df_train.groupby(out_label)[cat_columns[index]].sum())

  #print(df_train.groupby(out_label)[cat_columns[index]].value_counts())

  df_train.groupby(out_label)[cat_columns[index]].sum().plot.bar()

  plt.title(cat_columns[index])

  #sns.barplot(x= cat_columns[index], y=out_label, data=df_train)
for index in range(len(cat_columns)):

  plt.figure(index, figsize=(10,5))

  df_train.groupby(out_label)[cat_columns[index]].value_counts().plot.bar()

  plt.title(cat_columns[index])
for col in df_train.columns:

  print( col , " -> " , len(df_train[col].value_counts()) )

  #print( df_train[col].value_counts() )
def remove_outliers(df, col):

  q1 = df[col].quantile(0.25)

  q3 = df[col].quantile(0.75)

  iqr = q3 - q1

  min_threshold = q1 - 1.5*iqr

  max_threshold = q3 + 1.5*iqr

  #min_threshold, max_threshold = df[col].quantile([0.01,0.99])

  print(" Column Name ->", col, "\n min-threshold -> " , min_threshold, "\n max-threshold -> " , max_threshold )

  print("#######################################################################################################")

  df.loc[df[col]>=max_threshold , col] = df[col].mean()

  df.loc[df[col]<=min_threshold , col] = df[col].mean()

  #df.loc[df[col]>=max_threshold , col] = max_threshold ## capping the thresholds

  #df.loc[df[col]<=min_threshold , col] = min_threshold

  #df[col].plot.box()

  return df
df_train.corr()
df_train.head()
df_train = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
df_train = df_train.drop(columns=['Soil_Type7', 'Soil_Type15', 'Id'])
df_train.columns
con_columns
#for col in con_columns:

#  df_train = remove_outliers(df_train, col)
df_train[con_columns].describe()
df_test.head()
for col in df_test.columns:

  print( col , " -> " , len(df_test[col].value_counts()) )
df_test = df_test.drop(columns=['Soil_Type7', 'Soil_Type15', 'Id'])
df_test.columns
from sklearn.preprocessing import MinMaxScaler





X_train = df_train.drop(columns=['Cover_Type'])

y_train = df_train["Cover_Type"]

#X_train = df_train



X_test = df_test



# fit scaler on training data

norm = MinMaxScaler().fit(X_train)



# transform training data

X_train_norm = norm.transform(X_train)



#y_train = X_train_norm["Cover_Type"] 

#X_train_norm = X_train_norm.drop(columns=['Cover_Type'])



# transform testing data

X_test_norm = norm.transform(X_test)
def ml_algorithm(algo, X_train, y_train, cv):



  model = algo.fit(X_train, y_train)

  acc = round(model.score(X_train, y_train)* 100, 2)



  train_pred = model_selection.cross_val_predict(algo, 

                                                  X_train, 

                                                  y_train, 

                                                  cv=cv, 

                                                  n_jobs = -1)

  

  acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)

    

  return train_pred, acc, acc_cv
start_time = time.time()

train_pred_log, acc_log, acc_cv_log = ml_algorithm(LogisticRegression(), 

                                                               X_train_norm, 

                                                               y_train, 

                                                                    10)

log_time = (time.time() - start_time)

print("Accuracy: %s" % acc_log)

print("Accuracy CV 10-Fold: %s" % acc_cv_log)

print("Running Time: %s" % datetime.timedelta(seconds=log_time))
# k-Nearest Neighbours

start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = ml_algorithm(KNeighborsClassifier(), 

                                                  X_train_norm, 

                                                  y_train, 

                                                  10)

knn_time = (time.time() - start_time)

print("Accuracy: %s" % acc_knn)

print("Accuracy CV 10-Fold: %s" % acc_cv_knn)

print("Running Time: %s" % datetime.timedelta(seconds=knn_time))
# Gaussian Naive Bayes

start_time = time.time()

train_pred_gaussian, acc_gaussian, acc_cv_gaussian = ml_algorithm(GaussianNB(), 

                                                                      X_train_norm, 

                                                                      y_train, 

                                                                           10)

gaussian_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gaussian)

print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)

print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))
# Linear SVC

start_time = time.time()

train_pred_svc, acc_linear_svc, acc_cv_linear_svc = ml_algorithm(LinearSVC(),

                                                                X_train_norm, 

                                                                y_train, 

                                                                10)

linear_svc_time = (time.time() - start_time)

print("Accuracy: %s" % acc_linear_svc)

print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)

print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))
# Stochastic Gradient Descent

start_time = time.time()

train_pred_sgd, acc_sgd, acc_cv_sgd = ml_algorithm(SGDClassifier(), 

                                                  X_train_norm, 

                                                  y_train,

                                                  10)

sgd_time = (time.time() - start_time)

print("Accuracy: %s" % acc_sgd)

print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)

print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))
start_time = time.time()

train_pred_gbt, acc_gbt, acc_cv_gbt = ml_algorithm(GradientBoostingClassifier(), 

                                                                       X_train_norm, 

                                                                       y_train,

                                                                       10)

gbt_time = (time.time() - start_time)

print("Accuracy: %s" % acc_gbt)

print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)

print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))
model = GradientBoostingClassifier(random_state=45).fit(X_train_norm, y_train)



acc = round(model.score(X_train_norm, y_train)* 100, 2) 



pred = model.predict(X_test_norm)

print("Accuracy of the model is {}".format(acc))
print(pred.shape)
unique, counts = np.unique(pred, return_counts=True)

print(dict(zip(unique, counts)))
print(type(X_test_norm))

print(X_test_norm.shape)
df_test = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")
df_test.columns
submission = pd.DataFrame()

submission['Id'] = df_test['Id']

submission['Cover_Type'] = pred # our model predictions on the test dataset

submission.head(10)
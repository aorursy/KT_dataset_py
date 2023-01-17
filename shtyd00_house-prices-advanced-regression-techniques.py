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
#load packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

from sklearn import metrics

print("scikit-learn version: {}". format(sklearn.__version__))



import math

#print("math version: {}". format(math.__version__))



import seaborn as sns

print("seaborn version: {}". format(sns.__version__))



#import tensorflow as tf

#from tensorflow.python.data import Dataset

#print("tensorflow version: {}". format(tf.__version__))



#tf.logging.set_verbosity(tf.logging.ERROR)

#pd.options.display.max_columns = 15

#pd.options.display.max_rows = 15

pd.options.display.max_info_columns = 300

pd.options.display.float_format = '{:.1f}'.format
train_dataframe = pd.read_csv("../input/train.csv",sep=",")

test_dataframe = pd.read_csv("../input/test.csv",sep=",")

original_test_dataframe = test_dataframe



train_dataframe = train_dataframe.reindex(np.random.permutation(train_dataframe.index))

#test_dataframe = test_dataframe.reindex(np.random.permutation(test_dataframe.index))



train_dataframe.head()

train_dataframe["SalePrice"].describe()
sns.distplot(train_dataframe["SalePrice"])
#skewness and kurtosis

print("Skewness: %f" % train_dataframe['SalePrice'].skew())

print("Kurtosis: %f" % train_dataframe['SalePrice'].kurt())
ig, ax = plt.subplots(figsize=(24, 18)) 

sns.heatmap(train_dataframe.corr(), square=True, vmax=1, vmin=-1, center=0)
train_dataframe.info()
train_dataframe.describe()
# Function for nullanalysis

def nullAnalysis(df):

    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})



    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)

                         .T.rename(index={0:'null values (%)'}))

    return tab_info



# Show the null values

nullAnalysis(train_dataframe)

#convert categorical variable into dummy

# dummyに値はtrain set, test set合わせないといけない。

#print("train size = " % (train_dataframe.size))

print("train_dataframe.shape = " + str(train_dataframe.shape))

print("test_dataframe.shape = " + str(test_dataframe.shape))



all_dataframe = train_dataframe.append(test_dataframe, sort=False)

print("all_dataframe.shape = " + str(all_dataframe.shape))



all_dataframe_encoded = pd.get_dummies(all_dataframe)

print("all_dataframe_encoded.shape = " + str(all_dataframe_encoded.shape))



train_dataframe = all_dataframe_encoded[:1460]

test_dataframe = all_dataframe_encoded[1460:]

print("[encoded] train_dataframe.shape = " + str(train_dataframe.shape))

print("[encoded] test_dataframe.shape = " + str(test_dataframe.shape))



#drop caterogical variable

#train_dataframe = train_dataframe.select_dtypes(exclude=[object])

#train_dataframe.info()
train_dataframe = train_dataframe.drop(columns="LotFrontage")

train_dataframe = train_dataframe.drop(columns="MasVnrArea")

train_dataframe = train_dataframe.drop(columns="GarageYrBlt")

train_dataframe.info()
cor_saleprice = train_dataframe.corr()["SalePrice"] >= 0.3

#print(cor_saleprice)



train_dataframe = train_dataframe.loc[:, cor_saleprice]

print(train_dataframe.shape)

train_dataframe.info()
ig, ax = plt.subplots(figsize=(24, 18)) 

sns.heatmap(train_dataframe.corr(), square=True, vmax=1, vmin=-1, center=0)
len(train_dataframe)
train_dataframe_len = len(train_dataframe)

training_size = int(train_dataframe_len * 0.8)



train_target = train_dataframe["SalePrice"]

train_features = train_dataframe.drop(columns="SalePrice")



training_target = train_target[:training_size]

training_features = train_features[:training_size]

validation_target = train_target[training_size:]

validation_features = train_features[training_size:]
#Fit and Predict

#from xgboost import XGBClassifier

#xgb = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100)

#xgb.fit(train_features, train_target)

#y_test_xgb = xgb.predict(x_test)



import xgboost as xgb



# モデルにインスタンス生成

model = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)

model.fit(training_features, training_target)

training_predictions = model.predict(training_features)

validation_predictions = model.predict(validation_features)



#Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. 

training_rmse = np.sqrt(metrics.mean_squared_error(np.log(training_target), np.log(training_predictions)))

validation_rmse = np.sqrt(metrics.mean_squared_error(np.log(validation_target), np.log(validation_predictions)))

print ("training RMSE = " + str(training_rmse))

print ("validation RMSE = " + str(validation_rmse))
#test_dataframe = pd.read_csv("../input/test.csv",sep=",")

#test_dataframe = test_dataframe.reindex(np.random.permutation(test_dataframe.index))

#test_dataframe.info()



#convert categorical variable into dummy

#test_dataframe.info()

#test_dataframe = pd.get_dummies(test_dataframe)

#print("columns len = %d" % len(test_dataframe.columns))

#test_dataframe.info()



test_dataframe = test_dataframe.drop(columns="LotFrontage")

test_dataframe = test_dataframe.drop(columns="MasVnrArea")

test_dataframe = test_dataframe.drop(columns="GarageYrBlt")

test_dataframe = test_dataframe.drop(columns="SalePrice")

test_dataframe.info()





#相関の低いデータを落とす

test_dataframe = test_dataframe.loc[:, cor_saleprice]

#drop caterogical variable

#test_dataframe = test_dataframe.select_dtypes(exclude=[object])



#test_dataframe.info()



#drop missing features

#test_dataframe = test_dataframe.drop(columns="LotFrontage")

#test_dataframe = test_dataframe.drop(columns="MasVnrArea")

#test_dataframe = test_dataframe.drop(columns="GarageYrBlt")



#test_dataframe.info()
predictions = model.predict(test_dataframe)



#test_dataframe["SalePrice"] = predictions

original_test_dataframe["SalePrice"] = predictions



#print(test_dataframe[["Id", "SalePrice"]].shape)

#test_dataframe[["Id", "SalePrice"]].to_csv('submission.csv', index=False)
print(original_test_dataframe[["Id", "SalePrice"]].shape)

print(original_test_dataframe[["Id", "SalePrice"]].head())

original_test_dataframe[["Id", "SalePrice"]].to_csv('submission.csv', index=False)
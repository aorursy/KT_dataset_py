# import all libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

import re



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.preprocessing import scale

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline



import warnings # supress warnings

warnings.filterwarnings('ignore')
# Loading the data

dataP1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')

dataW1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')

dataP2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Generation_Data.csv')

dataW2 = pd.read_csv('../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv')





dataP1.head(5)
len(dataP1)
dataP1.hist(figsize=(15,12),bins = 15)

plt.title("Features Distribution")

plt.show()
dataP2.hist(figsize=(15,12),bins = 15)

plt.title("Features Distribution")

plt.show()
dataW1.hist(figsize=(15,12),bins = 15)

plt.title("Features Distribution")

plt.show()
dataP1.hist(figsize=(15,12),bins = 15)

plt.title("Features Distribution")

plt.show()
import numpy as np



from sklearn.model_selection import train_test_split

p1_train, p1_test = train_test_split(dataP1, test_size = 0.1, random_state = 10 ,shuffle=False )

p2_train, p2_test = train_test_split(dataP2, test_size = 0.1, random_state = 10 ,shuffle=False )

w1_train, w1_test = train_test_split(dataW1, test_size = 0.1, random_state = 10 ,shuffle=False )

w2_train, w2_test = train_test_split(dataW2, test_size = 0.1, random_state = 10 ,shuffle=False )



p1_train.tail(5)
p1_test.head(5)
print(p1_test)
p1_train = p1_train.drop(['PLANT_ID','SOURCE_KEY'],axis=1)

p1_test = p1_test.drop(['PLANT_ID','SOURCE_KEY'],axis=1) 

p2_train = p2_train.drop(['PLANT_ID','SOURCE_KEY'],axis=1)

p2_test = p2_test.drop(['PLANT_ID','SOURCE_KEY'],axis=1) 

w1_train = w1_train.drop(['PLANT_ID','SOURCE_KEY'],axis=1)

w1_test = w1_test.drop(['PLANT_ID','SOURCE_KEY'],axis=1) 

w2_train = w2_train.drop(['PLANT_ID','SOURCE_KEY'],axis=1)

w2_test = w2_test.drop(['PLANT_ID','SOURCE_KEY'],axis=1)
print(p1_test)
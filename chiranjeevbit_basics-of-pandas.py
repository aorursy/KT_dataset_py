import xgboost
import numpy as np
import pandas as pd
from math import sqrt

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
train = pd.read_csv("../input/train_4aqQp50.csv")
test = pd.read_csv("../input/test_VJP2kVH.csv")
train.head()
train.info()
train

train.describe()
train.columns

train.values
train['INT_SQFT'].plot()
train['INT_SQFT']
train.AREA.value_counts()
#replacing errors to the corresponding values
train.AREA.replace(['Chrompt', 'Chrmpet','Chormpet','TNagar', 'Ana Nagar','Ann Nagar', 'Karapakam', 'Velchery', 'KKNagar', 'Adyr'], \
['Chrompet', 'Chrompet', 'Chrompet','T Nagar', 'Anna Nagar', 'Anna Nagar', 'Karapakkam', 'Velachery', 'KK Nagar', 'Adyar'], inplace=True)
train.AREA.value_counts()
test.AREA.replace(['Velchery','Karapakam','Chrmpet','Ann Nagar', 'Chormpet', 'Chrompt'],\
['Velachery','Karapakkam','Chrompet','Anna Nagar','Chrompet','Chrompet'], inplace=True)

train.SALE_COND.replace(['Ab Normal', 'Adj Land','PartiaLl','Partiall'], \
['AbNormal', 'AdjLand', 'Partial','Partial'], inplace=True)

test.SALE_COND.replace(['Adj Land','PartiaLl','Partiall'], \
['AdjLand', 'Partial','Partial'], inplace=True)

train.PARK_FACIL.replace(['Noo'], \
['No'], inplace=True)

test.PARK_FACIL.replace(['Noo'], \
['No'], inplace=True)

train.BUILDTYPE.replace(['Other','Comercial'], \
['Others','Commercial'], inplace=True)

test.BUILDTYPE.replace(['Other','Comercial', 'Commercil'], \
['Others','Commercial', 'Commercial'], inplace=True)

train.UTILITY_AVAIL.replace(['All Pub'], \
['AllPub'], inplace=True)

test.UTILITY_AVAIL.replace(['All Pub'], \
['AllPub'], inplace=True)

train.STREET.replace(['Pavd', 'NoAccess'], \
['Paved', 'No Access'], inplace=True)

test.STREET.replace(['Pavd', 'NoAccess'], \
['Paved', 'No Access'], inplace=True)
train.AREA.value_counts()
train.info()
#Dropping table
train = train.drop(['PRT_ID', 'DATE_SALE', 'DATE_BUILD'], axis=1)
# we assign
#verticle axis = 1
#horizontal axis = 0

train  = train.fillna(0)
#train = train.dropna(0)
train.info()
train.describe()
scale_list = ['INT_SQFT','DIST_MAINROAD','REG_FEE','COMMIS','QS_ROOMS','QS_BATHROOM','QS_BEDROOM','QS_OVERALL']
sc = train[scale_list]
sc.head()
scaler = StandardScaler()
sc = scaler.fit_transform(sc)

train[scale_list] = sc

train[scale_list].head()
sc[0]
sc.head()
encoding_list = ['AREA', 'SALE_COND','PARK_FACIL','BUILDTYPE','UTILITY_AVAIL', 'STREET','MZZONE']
train[encoding_list] = train[encoding_list].apply(LabelEncoder().fit_transform)
train.head()
y = train['SALES_PRICE']
x = train.drop('SALES_PRICE', axis=1)

y.head()
x.head()
X_train, X_test, y_train, y_test = train_test_split(x, y ,test_size=0.3)

X_train.shape
X_test.shape
y_test.shape
logreg=LinearRegression()
#training
logreg.fit(X_train,y_train)
LinearRegression().fit(X_train,y_train)
y_pred=logreg.predict(X_test)
y_test
y_pred[0:6]
print(metrics.mean_squared_error(y_test, y_pred))
xgb = xgboost.XGBRegressor(n_estimators=25000, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
xgb.fit(X_train,y_train)
predictions = xgb.predict(X_test)

print(sqrt(metrics.mean_squared_error(y_test, predictions)))

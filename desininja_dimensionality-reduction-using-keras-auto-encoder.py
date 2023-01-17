import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model
import xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.pyplot as plote

from sklearn import cross_validation, metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train2 = train.copy()
train3 = train.copy()
#target = train['target']
#train_id = train['ID']
#test_id = test['ID']

#train.drop(['target'], axis=1, inplace=True)
#train.drop(['ID'], axis=1, inplace=True)
#test.drop(['ID'], axis=1, inplace=True)
print('Train data shape', X_train.shape)
print('Test data shape', X_test.shape)
#train_scaled = minmax_scale(train, axis = 0)
#test_scaled = min3max_scale(test, axis = 0)
scale_list = train3.columns[1:]
sc = train3[scale_list]
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
train3[scale_list] = sc
train3[scale_list].head()


# define the number of features
ncol = X_train.shape[1]
ncol
#X_train, X_test, Y_train, Y_test = train_test_split(train_scaled, target, train_size = 0.9, random_state = seed(2017))

X3 = train3.drop(['target','ID'], axis=1)
Y3 = train3['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.2)


### Define the encoder dimension
encoding_dim = 200
input_dim = Input(shape = (ncol, ))

# Encoder Layers
encoded1 = Dense(3000, activation = 'relu')(input_dim)
encoded2 = Dense(2750, activation = 'relu')(encoded1)
encoded3 = Dense(2500, activation = 'relu')(encoded2)
encoded4 = Dense(2250, activation = 'relu')(encoded3)
encoded5 = Dense(2000, activation = 'relu')(encoded4)
encoded6 = Dense(1750, activation = 'relu')(encoded5)
encoded7 = Dense(1500, activation = 'relu')(encoded6)
encoded8 = Dense(1250, activation = 'relu')(encoded7)
encoded9 = Dense(1000, activation = 'relu')(encoded8)
encoded10 = Dense(750, activation = 'relu')(encoded9)
encoded11 = Dense(500, activation = 'relu')(encoded10)
encoded12 = Dense(250, activation = 'relu')(encoded11)
encoded13 = Dense(encoding_dim, activation = 'relu')(encoded12)

# Decoder Layers
decoded1 = Dense(250, activation = 'relu')(encoded13)
decoded2 = Dense(500, activation = 'relu')(decoded1)
decoded3 = Dense(750, activation = 'relu')(decoded2)
decoded4 = Dense(1000, activation = 'relu')(decoded3)
decoded5 = Dense(1250, activation = 'relu')(decoded4)
decoded6 = Dense(1500, activation = 'relu')(decoded5)
decoded7 = Dense(1750, activation = 'relu')(decoded6)
decoded8 = Dense(2000, activation = 'relu')(decoded7)
decoded9 = Dense(2250, activation = 'relu')(decoded8)
decoded10 = Dense(2500, activation = 'relu')(decoded9)
decoded11 = Dense(2750, activation = 'relu')(decoded10)
decoded12 = Dense(3000, activation = 'relu')(decoded11)
decoded13 = Dense(ncol, activation = 'sigmoid')(decoded12)

# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = decoded13)

# Compile the Model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')
autoencoder.summary()
autoencoder.fit(X_train, X_train, nb_epoch = 10, batch_size = 32, shuffle = False, validation_data = (X_test, X_test))
encoder = Model(inputs = input_dim, outputs = encoded13)
encoded_input = Input(shape = (encoding_dim, ))
encoded_train = pd.DataFrame(encoder.predict(X_train))
encoded_train = encoded_train.add_prefix('feature_')

encoded_test = pd.DataFrame(encoder.predict(X_test))
encoded_test = encoded_test.add_prefix('feature_')
print(encoded_train.shape)
print(encoded_train.shape)
encoded_train.head(5)
print(encoded_test.shape)
encoded_test.head()
encoded_train.to_csv('train_encoded.csv', index=False)
encoded_test.to_csv('test_encoded.csv', index=False)
encoded_test = encoded_test.fillna(0)
sns.heatmap(encoded_train.isnull(),yticklabels = False,cbar = False,cmap = 'viridis')
missing_val_count_by_column = (encoded_test.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
#encoder + PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
x_train = pca.fit_transform(encoded_train)
x_test = pca.transform(encoded_test)
explained_variance = pca.explained_variance_ratio_
explained_variance
xgb = xgboost.XGBRegressor(n_estimators=35, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
                           
                           
xgb.fit(x_train,y_train)
predictions = xgb.predict(x_test)
print(metrics.mean_squared_error(y_test, predictions))
rand = RandomForestRegressor(n_estimators = 10,random_state = 0)
rand.fit(x_train,y_train)
y_pred2 = rand.predict(x_test)
print(metrics.mean_squared_error(y_test,y_pred2 ))

logreg=LinearRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

y_pred
print(metrics.mean_squared_error(y_test, y_pred))

regressor = DecisionTreeRegressor( random_state = 0)
regressor.fit(x_train,y_train)
y_pred1 = regressor.predict(x_test)
print(metrics.mean_squared_error(y_test,y_pred1 ))
scale_list = train2.columns[1:]
sc = train2[scale_list]
scaler = StandardScaler()
sc = scaler.fit_transform(sc)
train2[scale_list] = sc
train2[scale_list].head()

X = train2.drop(['target','ID'], axis=1)
Y = train2['target']
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.2)
#PCA ONLY
from sklearn.decomposition import PCA
pca = PCA(n_components=None)
x2_train = pca.fit_transform(X_train)
x2_test = pca.transform(X_test)
explained_variance2 = pca.explained_variance_ratio_
regressor = DecisionTreeRegressor( random_state = 0)
regressor.fit(x2_train,y_train)
y_pred1 = regressor.predict(x2_test)
print(metrics.mean_squared_error(y_test,y_pred1 ))
xgb = xgboost.XGBRegressor(n_estimators=35, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
                           
                           
xgb.fit(x2_train,y_train)
predictions = xgb.predict(x2_test)
print(metrics.mean_squared_error(y_test, predictions))
rand = RandomForestRegressor(n_estimators = 10,random_state = 0)
rand.fit(x2_train,y_train)
y_pred2 = rand.predict(x2_test)
print(metrics.mean_squared_error(y_test,y_pred2 ))

logreg=LinearRegression()
logreg.fit(x2_train,y_train)
y_pred=logreg.predict(x2_test)

y_pred
print(metrics.mean_squared_error(y_test, y_pred))

#KERNEL PCA + ENCODER
from sklearn.decomposition import  KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
x4_train = kpca.fit_transform(encoded_train)
x4_test = kpca.transform(encoded_test)
regressor = DecisionTreeRegressor( random_state = 0)
regressor.fit(x4_train,y_train)
y_pred1 = regressor.predict(x4_test)
print(metrics.mean_squared_error(y_test,y_pred1 ))
xgb = xgboost.XGBRegressor(n_estimators=35, learning_rate=0.06, gamma=0, subsample=0.6,
                           colsample_bytree=0.7, min_child_weight=4, max_depth=3)
                           
                           
xgb.fit(x4_train,y_train)
predictions = xgb.predict(x4_test)
print(metrics.mean_squared_error(y_test, predictions))
rand = RandomForestRegressor(n_estimators = 10,random_state = 0)
rand.fit(x4_train,y_train)
y_pred2 = rand.predict(x4_test)
print(metrics.mean_squared_error(y_test,y_pred2 ))
logreg=LinearRegression()
logreg.fit(x4_train,y_train)
y_pred=logreg.predict(x4_test)

y_pred
print(metrics.mean_squared_error(y_test, y_pred))



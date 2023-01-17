#loading need libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
#Process Data: Categorize inputs, remove unuseful data, fill missing values, encoding some string values...
def process_data(train):

	categories = ['currency', 'property_type', 'place_name', 'state_name']
	for cat in categories:
	    train[cat] = pd.Categorical(train[cat], categories=train[cat].unique()).codes

	#Correlation between train attributes

	#Separate variable into new dataframe from original dataframe which has only numerical values
	train_corr = train.select_dtypes(include=[np.number])
	del train_corr['id']
	del train_corr['geonames_id']

	train = train_corr

	train['floor'] = train['floor'].fillna('None') #high missing ratio
	train['surface_total_in_m2'] = train['surface_total_in_m2'].fillna('None')
	train['expenses'] = train['expenses'].fillna('None')
	train['lon'] = train.groupby('place_name')['lon'].transform
	(
	    lambda x: x.fillna(x.median()))
	train['lat'] = train.groupby('place_name')['lat'].transform
	(
	    lambda x: x.fillna(x.median()))
	train['rooms'] = train['rooms'].fillna(train['rooms'].median())
	train['surface_covered_in_m2'] = train['surface_covered_in_m2'].fillna(train['surface_covered_in_m2'].median())

	from sklearn.preprocessing import LabelEncoder
	cols = ['property_type', 'place_name', 'state_name', 'lat', 'lon', 'currency', 'surface_total_in_m2', 
	        'surface_covered_in_m2', 'floor', 'rooms', 'expenses']
	for c in cols:
	    lbl = LabelEncoder() 
	    lbl.fit(list(train[c].values)) 
	    train[c] = lbl.transform(list(train[c].values))
	return train
def transform_outliers(train):
	cols = ['floor', 'expenses','surface_covered_in_m2']#, 'surface_total_in_m2']
	for col in cols:
	    q = train[col].quantile(0.99)
	    train.loc[train[col] > q, col] = q
	return train
def scale_data(train):
	scaler = StandardScaler()
	train = scaler.fit_transform(train)
	return pd.DataFrame(train), scaler
#Read data and prepare 
train = pd.read_csv('features-training.csv')
test = pd.read_csv('target-training.csv')

test['price'] = np.log1p(test['price'])
train = process_data(train)
train = transform_outliers(train)

train, scaler = scale_data(train)


train = pd.concat([train,test['price']], axis=1)
y = train['price']
del train['price']
X = train.values
y = y.values
#Import Keras 
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model, Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.callbacks import LearningRateScheduler, TensorBoard
from sklearn.metrics import mean_squared_error
#Hyperparameters
filename = 'house_prediction'
ACTIVATION = 'relu'
EPOCHS = 2
BATCH_SIZE = 128
LEARNING_RATE= 0.01
LOSS_FUNCTION = 'mean_squared_error'
BETA_1 = 0.9
BETA_2 = 0.999
EPSILON = 1e-08
EPOCHS_DROP = 20
DROP = 0.8
#Build Neural Network
model =  Sequential()
model.add(Dense(128,  input_dim=13,  activation = ACTIVATION))
model.add(Dense(128, activation = ACTIVATION))
model.add(Dense(64,  activation = ACTIVATION))
model.add(Dense(1))
#Define loss metric
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
#Define learning rate decay callback
def step_decay(epoch):
   initial_lrate = LEARNING_RATE
   drop = DROP
   epochs_drop = float(EPOCHS_DROP)
   lrate = initial_lrate * np.power(drop,  
           np.floor((1+epoch)/epochs_drop))
   if epoch % 10 == 0:
   		model.save(filename + "_graph_" + str(epoch))
   return lrate
#Fit and predict model (longest part...)
lrate = LearningRateScheduler(step_decay)
tbCallBack = TensorBoard(log_dir='./' + filename, histogram_freq=0, write_graph=True, write_images=True)

adam = Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, epsilon=EPSILON)
model.compile (loss = root_mean_squared_error, optimizer = adam, metrics = ['mse'])
model.fit (X, y, epochs = EPOCHS, batch_size = BATCH_SIZE, verbose = 1, callbacks=[lrate, tbCallBack], validation_split=0.1)

y_hat = model.predict(X)
print ("Total LMSE: " + str(mean_squared_error(y_hat, y)))

y_hat = np.expm1(y_hat)
np.savetxt('y_hat.csv', y_hat, delimiter=',', fmt=['%.10f'])
#Predict lb data
lb = pd.read_csv('features-test.csv')
id_lb = lb['id'].values.reshape((lb['id'].values.shape[0], 1)).astype(int)
lb = process_data(lb)
lb = transform_outliers(lb)
lb = pd.DataFrame(scaler.transform(lb))
y_lb = np.expm1(model.predict(lb.values))

print (id_lb)
res = np.hstack([id_lb, y_lb])
np.savetxt('submission.csv', res, delimiter=',', fmt=['%d', '%.10f'])


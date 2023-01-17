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
import numpy as np 

import pandas as pd



from sklearn.preprocessing import MinMaxScaler



from keras.models import Sequential, load_model

from keras.layers import *
import os 

inputFolder = '../input/' 

for root, directories, filenames in os.walk(inputFolder): 

    for filename in filenames: print(os.path.join(root,filename))
training_data_df = pd.read_csv(r'../input/sample-data/sales_data_training.csv',sep=',')
#data Exploratiton

training_data_df.shape
test_data_df =pd.read_csv(r'../input/sample-data/sales_data_test.csv',sep=',')
test_data_df.shape
scaler = MinMaxScaler(feature_range = (0,1))
scaled_training_df = scaler.fit_transform(training_data_df)

scaled_test_df = scaler.transform(test_data_df)

print('Note:total_earnings values were scaled by multiplying by {:.10f} and adding {:.6f}'.format(scaler.scale_[8],scaler.min_[8]))
scaled_training_df = pd.DataFrame(scaled_training_df,columns=training_data_df.columns)

scaled_test_df = pd.DataFrame(scaled_test_df,columns=test_data_df.columns)

X = scaled_training_df.drop('total_earnings',axis=1).values

Y = scaled_training_df['total_earnings'].values



#defining the model

model = Sequential()

model.add(Dense(50,input_dim=9, activation= 'relu',name='layer_1'))

model.add(Dense(100, activation='relu',name='layer_2'))

model.add(Dense(50, activation='relu',name='layer_3'))

model.add(Dense(1, activation= 'linear',name='outer_layer'))

model.compile(loss='mean_squared_error',optimizer = 'adam')

# training the model

model.fit(X,Y,epochs=100, shuffle=True, verbose=2)
X_test = scaled_test_df.drop('total_earnings',axis = 1).values

Y_test = scaled_test_df['total_earnings'].values
test_error_rate = model.evaluate(X_test,Y_test, verbose=0)



print("the mean Squared error (MSE)for the test data is: {}".format(test_error_rate))
df_test = pd.read_csv(r'../input/sample-data/proposed_new_product.csv',sep=',').values
prediction = model.predict(df_test)

prediction = prediction[0][0]

prediction = (prediction +0.1159)/0.0000036968

prediction 
#Image Recognization
from keras.preprocessing import image

from keras.applications.resnet50 import ResNet50

from keras.applications.resnet50 import preprocess_input, decode_predictions
model = ResNet50(weights='imagenet')
img = image.load_img('../input/image-recog/car.jpg',target_size=(224,224))
x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

x = preprocess_input(x)
prediction = model.predict(x)
prediction_classes = decode_predictions(prediction, top=9)
for imagenet_id, name, likelihood in prediction_classes[0]:

    print('-{}:{:2f} likelihood'.format(name,likelihood))
#logger 

import pandas as pd

import keras

from keras.models import Sequential

from keras.layers import *

logger = keras.callbacks.TensorBoard(log_dir='logs', write_graph = True,

                                    histogram_freq =0)
#defining the model

model = Sequential()

model.add(Dense(50,input_dim=9, activation= 'relu',name='layer_1'))

model.add(Dense(100, activation='relu',name='layer_2'))

model.add(Dense(50, activation='relu',name='layer_3'))

model.add(Dense(1, activation= 'linear',name='outer_layer'))

model.compile(loss='mean_squared_error',optimizer = 'adam')
model.fit(X,Y,

          epochs=50,

          shuffle=True,

          verbose=2,

          callbacks=[logger]

         )
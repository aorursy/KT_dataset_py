# %matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.utils import np_utils

from pandas import read_csv

from pandas import DataFrame

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
def l2_dist(p1, p2):

    x1,y1 = p1

    x2,y2 = p2

    x1, y1 = np.array(x1), np.array(y1)

    x2, y2 = np.array(x2), np.array(y2)

    dx = x1 - x2

    dy = y1 - y2

    dx = dx ** 2

    dy = dy ** 2

    dists = dx + dy

    dists = np.sqrt(dists)

    return np.mean(dists), dists
path='../input/iBeacon_RSSI_Labeled.csv'

x = read_csv(path, index_col=None)
x.head(5)
for col in x.columns[10:]:

    x.hist(column = col)
def fix_pos(x_cord):

    x = 87 - ord(x_cord.upper())

    return x
path='../input/iBeacon_RSSI_Labeled.csv'

x = read_csv(path, index_col=None)

x['x'] = x['location'].str[0]

x['y'] = x['location'].str[1:]

x.drop(["location"], axis = 1, inplace = True)

x["x"] = x["x"].apply(fix_pos)

x["y"] = x["y"].astype(int)
y = x.iloc[:, -2:]

x = x.iloc[:, 1:-2]

train_x, val_x, train_y, val_y = train_test_split(x,y, test_size = .2, shuffle = False)
from keras.optimizers import Adam

from keras.layers import BatchNormalization

from keras.callbacks import EarlyStopping

def create_deep(inp_dim):

    seed = 7

    np.random.seed(seed)

    model = Sequential()

    model.add(Dense(50, input_dim=inp_dim, activation='sigmoid'))

    model.add(BatchNormalization())

    model.add(Dense(50, activation='relu'))

    model.add(Dense(50, activation='relu'))

    model.add(Dense(2, activation='relu'))

    # Compile model

    model.compile(loss='mse', optimizer=Adam(.001), metrics=['mse'])

    return model
es = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='auto', restore_best_weights=True)

model = create_deep(train_x.shape[1])

hist = model.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=1000, batch_size=1000,  verbose=0, callbacks = [es])
preds = model.predict(val_x)

l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))

print(l2dists_mean)
sortedl2_deep = np.sort(l2dists)

prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)

fig, ax = plt.subplots()

lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')

plt.title('CDF of Euclidean distance error')

plt.xlabel('Distance (m)')

plt.ylabel('Probability')

plt.grid(True)

gridlines = ax.get_xgridlines() + ax.get_ygridlines()

for line in gridlines:

    line.set_linestyle('-.')



plt.savefig('Figure_CDF_error.png', dpi=300)

plt.show()

plt.close()
from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML

import numpy as np

from PIL import Image



image = Image.open("../input/iBeacon_Layout.jpg")

init_notebook_mode(connected=True)



xm=np.min(val_y["x"])-1.5

xM=np.max(val_y["x"])+1.5

ym=np.min(val_y["y"])-1.5

yM=np.max(val_y["y"])+1.5



data=[dict(x=[0], y=[0], 

           mode="markers", name = "Predictions",

           line=dict(width=2, color='green')

          ),

      dict(x=[0], y=[0], 

           mode="markers", name = "Actual",

           line=dict(width=2, color='blue')

          )

      

    ]



layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),

            yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),

            title='Moving Dots', hovermode='closest',

            images= [dict(

                  source= image,

                  xref= "x",

                  yref= "y",

                  x= -3.5,

                  y= 22,

                  sizex= 36,

                  sizey=25,

                  sizing= "stretch",

                  opacity= 0.5,

                  layer= "below")]

            )



frames=[dict(data=[dict(x=[preds[k, 0]], 

                        y=[preds[k, 1]], 

                        mode='markers',

                        

                        marker=dict(color='red', size=10)

                        ),

                   dict(x=[val_y["x"].iloc[k]], 

                        y=[val_y["y"].iloc[k]], 

                        mode='markers',

                        

                        marker=dict(color='blue', size=10)

                        )

                  ]) for k in range(int(len(preds))) 

       ]    

          

figure1=dict(data=data, layout=layout, frames=frames)          

iplot(figure1)
# data=[dict(x=[1, 1], y=[16, 16], 

#            mode="markers", name = "Predictions",

#            line=dict(width=2, color='green')

#           ),

#       dict(x=[2, 2], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[3, 3], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[4, 4], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[5, 5], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[6, 6], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[7, 7], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[8, 8], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[9, 9], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[10, 10], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[11, 11], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[12, 12], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[13, 13], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[14, 14], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[15, 15], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[16, 16], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[17, 17], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[18, 18], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[19, 19], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[20, 20], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[20, 20], y=[16, 16], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[20, 20], y=[17, 17], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[20, 20], y=[18, 18], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           ),

#       dict(x=[20, 20], y=[19, 19], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           )

#       ,

#       dict(x=[20, 20], y=[0, 0], 

#            mode="markers", name = "Actual",

#            line=dict(width=2, color='blue')

#           )

      

#     ]

# layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),

#             yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),

#             title='Moving Dots', hovermode='closest',

#             images= [dict(

#                   source= image,

#                   xref= "x",

#                   yref= "y",

#                   x= -3.5,

#                   y= 22,

#                   sizex= 36,

#                   sizey=25,

#                   sizing= "stretch",

#                   opacity= 0.5,

#                   layer= "below")]

#             )

# figure1=dict(data=data, layout=layout)          

# iplot(figure1)
path='../input/iBeacon_RSSI_Labeled.csv'

x = read_csv(path, index_col=None)

x['x'] = x['location'].str[0]

x['y'] = x['location'].str[1:]

x.drop(["location"], axis = 1, inplace = True)

x["x"] = x["x"].apply(fix_pos)

x["y"] = x["y"].astype(int)

y = x.iloc[:, -2:]

x = x.iloc[:, 1:-2]

x["b3001"].values.shape
img_x = np.zeros(shape = (x.shape[0], 25, 25, 1, ))

beacon_coords = {"b3001": (5, 9), 

                 "b3002": (9, 14), 

                 "b3003": (13, 14), 

                 "b3004": (18, 14), 

                 "b3005": (9, 11), 

                 "b3006": (13, 11), 

                 "b3007": (18, 11), 

                 "b3008": (9, 8), 

                 "b3009": (2, 3), 

                 "b3010": (9, 3), 

                 "b3011": (13, 3), 

                 "b3012": (18, 3), 

                 "b3013": (22, 3),}

for key, value in beacon_coords.items():

    img_x[:, value[0], value[1], 0] -= x[key].values/200

    print(key, value)

# img_x = (img_x) / 200

train_x, val_x, train_y, val_y = train_test_split(img_x, y, test_size = .2, shuffle = False)
img_x[103, :, :, 0].mean()
#what one sample looks like 

img_x[1, :, :, 0]
img = Image.fromarray(img_x[19, :, :, 0] * 255, "L")

img.resize((250, 250))
from keras.models import Model, Input

from keras.layers import Conv2D, MaxPooling2D, Flatten, Conv2DTranspose
from keras import backend as K

def rmse(y_true, y_pred):

        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 
inputs = Input(shape=(train_x.shape[1], train_x.shape[2], 1))



# a layer instance is callable on a tensor, and returns a tensor

x = Conv2D(3, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(inputs)

x = MaxPooling2D(2)(x)

x = Conv2D(6, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)

x = MaxPooling2D(2)(x)

x = Conv2D(12, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)

predictions = Dense(2, activation='relu')(Flatten()(x))



# This creates a model that includes

# the Input layer and three Dense layers

model = Model(inputs=inputs, outputs=predictions)

model.compile(optimizer=Adam(.001),

              loss=rmse,

              metrics=['accuracy'])

model.summary()

hist = model.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=1000, batch_size=200,  verbose=0, callbacks = [es])
preds = model.predict(val_x)

l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))

print(l2dists_mean)


plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
sortedl2_deep = np.sort(l2dists)

prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)

fig, ax = plt.subplots()

lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')

plt.title('CDF of Euclidean distance error')

plt.xlabel('Distance (m)')

plt.ylabel('Probability')

plt.grid(True)

gridlines = ax.get_xgridlines() + ax.get_ygridlines()

for line in gridlines:

    line.set_linestyle('-.')



plt.savefig('Figure_CDF_error.png', dpi=300)

plt.show()

plt.close()
from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML

import numpy as np

from PIL import Image



image = Image.open("../input/iBeacon_Layout.jpg")

init_notebook_mode(connected=True)



xm=np.min(val_y["x"])-1.5

xM=np.max(val_y["x"])+1.5

ym=np.min(val_y["y"])-1.5

yM=np.max(val_y["y"])+1.5



data=[dict(x=[0], y=[0], 

           mode="markers", name = "Predictions",

           line=dict(width=2, color='green')

          ),

      dict(x=[0], y=[0], 

           mode="markers", name = "Actual",

           line=dict(width=2, color='blue')

          )

      

    ]



layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),

            yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),

            title='Moving Dots', hovermode='closest',

            images= [dict(

                  source= image,

                  xref= "x",

                  yref= "y",

                  x= -3.5,

                  y= 22,

                  sizex= 36,

                  sizey=25,

                  sizing= "stretch",

                  opacity= 0.5,

                  layer= "below")]

            )



frames=[dict(data=[dict(x=[preds[k, 0]], 

                        y=[preds[k, 1]], 

                        mode='markers',

                        

                        marker=dict(color='red', size=10)

                        ),

                   dict(x=[val_y["x"].iloc[k]], 

                        y=[val_y["y"].iloc[k]], 

                        mode='markers',

                        

                        marker=dict(color='blue', size=10)

                        )

                  ]) for k in range(int(len(preds))) 

       ]    

          

figure1=dict(data=data, layout=layout, frames=frames)          

iplot(figure1)
path='../input/iBeacon_RSSI_Unlabeled.csv'

x_un = read_csv(path, index_col=None)

# x['x'] = x['location'].str[0]

# x['y'] = x['location'].str[1:]

x_un.drop(["location", "date"], axis = 1, inplace = True)

# x["x"] = x["x"].apply(fix_pos)

# x["y"] = x["y"].astype(int)

# y = x.iloc[:, -2:]

# x = x.iloc[:, 1:-2]

img_x = np.zeros(shape = (x_un.shape[0], 25, 25, 1, ))

beacon_coords = {"b3001": (5, 9), 

                 "b3002": (9, 14), 

                 "b3003": (13, 14), 

                 "b3004": (18, 14), 

                 "b3005": (9, 11), 

                 "b3006": (13, 11), 

                 "b3007": (18, 11), 

                 "b3008": (9, 8), 

                 "b3009": (2, 3), 

                 "b3010": (9, 3), 

                 "b3011": (13, 3), 

                 "b3012": (18, 3), 

                 "b3013": (22, 3),}

for key, value in beacon_coords.items():

    img_x[:, value[0], value[1], 0]  -= x_un[key].values/200

    print(key, value)

train_x_un, val_x_un = train_test_split(img_x, test_size = .2, shuffle = False)
from keras.layers import GaussianNoise, AveragePooling2D, SpatialDropout2D

inputs = Input(shape=(train_x.shape[1], train_x.shape[2], 1))



# a layer instance is callable on a tensor, and returns a tensor

x = Conv2D(24, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(inputs)

x = MaxPooling2D(2)(x)

x = Conv2D(24, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)

x = MaxPooling2D(2)(x)

x = Conv2D(24, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)

x = Conv2DTranspose(24, kernel_size=(3,3),strides = (2,2), activation='relu', padding = "valid", data_format="channels_last")(x)

x = Conv2DTranspose(16, kernel_size=(3,3),strides = (2,2), activation='relu', padding = "valid", data_format="channels_last")(x)

x = Conv2DTranspose(8, kernel_size=(3,3),strides = (2,2), activation='relu', padding = "valid", data_format="channels_last")(x)

x = Conv2DTranspose(1, kernel_size=(3,3), activation='relu', padding = "valid", data_format="channels_last")(x)



# This creates a model that includes

# the Input layer and three Dense layers

model2 = Model(inputs=inputs, outputs=x)

model2.compile(optimizer=Adam(.0001, clipnorm = .5, clipvalue = .5),

              loss='mse',

              metrics=['accuracy'])

model2.summary()

hist = model2.fit(x = train_x_un, y = train_x_un, validation_data = (val_x_un,val_x_un),epochs=15, batch_size=10, verbose=2, callbacks = [es])
predictions = Dense(8, activation='relu')(Flatten()(model2.layers[5].output))

predictions = Dense(2, activation = 'relu')(predictions)
model3 = Model(inputs=model2.input, outputs=predictions)

model3.summary()
for layer in model3.layers[:-2]:

    layer.trainable = False

model3.compile(optimizer=Adam(.001, clipnorm = .5, clipvalue = .5),

              loss=rmse,

              metrics=['accuracy'])
hist = model3.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=500, batch_size=50,  verbose=0, callbacks = [es])
preds = model3.predict(val_x)

l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))

print(l2dists_mean)


plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
for layer in model3.layers[:-2]:

    layer.trainable = True

model3.compile(optimizer=Adam(.001, clipnorm = .5, clipvalue = .5),

              loss=rmse,

              metrics=['accuracy'])

hist = model3.fit(x = train_x, y = train_y, validation_data = (val_x,val_y), epochs=500, batch_size=50,  verbose=0, callbacks = [es])

preds = model3.predict(val_x)

l2dists_mean, l2dists = l2_dist((preds[:, 0], preds[:, 1]), (val_y["x"], val_y["y"]))

print(l2dists_mean)


plt.plot(hist.history['acc'])

plt.plot(hist.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(hist.history['loss'])

plt.plot(hist.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
sortedl2_deep = np.sort(l2dists)

prob_deep = 1. * np.arange(len(sortedl2_deep))/(len(sortedl2_deep) - 1)

fig, ax = plt.subplots()

lg1, = ax.plot(sortedl2_deep, prob_deep, color='black')

plt.title('CDF of Euclidean distance error')

plt.xlabel('Distance (m)')

plt.ylabel('Probability')

plt.grid(True)

gridlines = ax.get_xgridlines() + ax.get_ygridlines()

for line in gridlines:

    line.set_linestyle('-.')



plt.savefig('Figure_CDF_error.png', dpi=300)

plt.show()

plt.close()
from plotly.offline import init_notebook_mode, iplot

from IPython.display import display, HTML

import numpy as np

from PIL import Image



image = Image.open("../input/iBeacon_Layout.jpg")

init_notebook_mode(connected=True)



xm=np.min(val_y["x"])-1.5

xM=np.max(val_y["x"])+1.5

ym=np.min(val_y["y"])-1.5

yM=np.max(val_y["y"])+1.5



data=[dict(x=[0], y=[0], 

           mode="markers", name = "Predictions",

           line=dict(width=2, color='green')

          ),

      dict(x=[0], y=[0], 

           mode="markers", name = "Actual",

           line=dict(width=2, color='blue')

          )

      

    ]



layout=dict(xaxis=dict(range=[xm, 24], autorange=False, zeroline=False),

            yaxis=dict(range=[ym, 21], autorange=False, zeroline=False),

            title='Moving Dots', hovermode='closest',

            images= [dict(

                  source= image,

                  xref= "x",

                  yref= "y",

                  x= -3.5,

                  y= 22,

                  sizex= 36,

                  sizey=25,

                  sizing= "stretch",

                  opacity= 0.5,

                  layer= "below")]

            )



frames=[dict(data=[dict(x=[preds[k, 0]], 

                        y=[preds[k, 1]], 

                        mode='markers',

                        

                        marker=dict(color='red', size=10)

                        ),

                   dict(x=[val_y["x"].iloc[k]], 

                        y=[val_y["y"].iloc[k]], 

                        mode='markers',

                        

                        marker=dict(color='blue', size=10)

                        )

                  ]) for k in range(int(len(preds))) 

       ]    

          

figure1=dict(data=data, layout=layout, frames=frames)          

iplot(figure1)
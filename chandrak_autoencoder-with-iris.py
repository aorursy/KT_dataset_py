# !pip install keras
import pandas as pd
import numpy as np
from keras.models import Sequential
# from keras.layers import Dense, Activation, Embedding, Merge, Flatten
from keras.layers import Dense, Activation, Embedding, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv("../input/Iris.csv")
data.head()
data.columns
# data.drop(['Id'],inpalce=True)
scaler = StandardScaler()
data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']] = scaler.fit_transform(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]) 
data.head()

train_x, test_x, train_y, test_y = train_test_split(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']],data['Species'],test_size=0.1, random_state=1)
train_x.head()
train_x.shape, test_x.shape, train_y.shape, test_y.shape
model = Sequential()
model.add(Dense(input_dim=train_x.shape[1],output_dim=12))
model.add(Activation('relu'))
model.add(Dense(output_dim=2))
model.add(Activation('relu'))
model.add(Dense(input_dim=train_x.shape[1],output_dim=12))
model.add(Activation('relu'))
model.add(Dense(output_dim=4))
model.add(Activation('relu'))

model.compile(optimizer='rmsprop', loss='mean_squared_error')

model.fit(train_x, train_x, nb_epoch=250, verbose=2)
def get_activations(model, layer, X_batch):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])
    return activations
model.summary()
from keras import backend as K 
hidden_op= get_activations(model=model, layer=2, X_batch=train_x)
op2=pd.DataFrame({'column1':hidden_op[0][:,0], 'column2':hidden_op[0][:,1] , 'column3':train_y} )
op2.head()
d = {'Iris-setosa': 'green', 'Iris-versicolor': 'red', 'Iris-virginica': 'black'}
op2['color'] = op2['column3'].map(d)
op2.head()
%matplotlib inline
# import matplotlib.pyplot as plt 
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(30, 80)
# ax.scatter(xs=op2['column1'], ys=op2['column2'], zs=op2['column3'],c=op2['color'])
plt.scatter(op2['column1'], op2['column2'],c=op2['color'].values)
hidden_op_test= get_activations(model=model, layer=2, X_batch=test_x)
op3=pd.DataFrame({'column1':hidden_op_test[0][:,0], 'column2':hidden_op_test[0][:,1] , 'column3':test_y} )
op3['color'] = op3['column3'].map(d)
plt.scatter(op3['column1'], op3['column2'],c=op3['color'].values)
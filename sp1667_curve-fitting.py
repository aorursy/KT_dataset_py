import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
import keras

from keras.regularizers import l1,l2

from sklearn.metrics import mean_squared_error as mse

from sklearn.preprocessing import PolynomialFeatures
%matplotlib notebook

from IPython.html import *

from ipywidgets import *

from IPython.display import display

plt.style.use('default')

#from matplotlib.gridspec import GridSpec

#from skimage import color as COL_x

from keras.optimizers import Adam,SGD
from IPython.core.display import display, HTML

display(HTML("""<style>

.output_wrapper, .output {

    height:auto !important;

    max-height:5500px;  /* your desired max-height here */

}

.output_scroll {

    box-shadow:none !important;

    webkit-box-shadow:none !important;

}

</style>"""))
def sse(y1,y2):

    return np.sum((y1-y2)**2)*1000



def poly(order,samples,random_state = 2):

    np.random.seed(random_state)

    ar = np.array(range(-samples//2,samples//2)) / samples

    out = np.zeros(len(ar))

    for i in range(1,order+1):

        out += np.sin((ar)**order) + np.cos(np.sin(ar**(np.random.choice(range(order))))) 

    out = out + (np.random.random() * 5)

    out = out / out.max()

    return ar, out
'''

#m1

opt = Adam(learning_rate=0.0001)

model = keras.models.Sequential()

model.add(keras.layers.Dense(500, input_dim = dim ,activation = "relu"))

model.add(keras.layers.Dense(500,activation = "relu"))

model.add(keras.layers.Dense(100,activation = "relu"))

model.add(keras.layers.Dense(1))



#m2

opt = Adam(learning_rate=0.0001)

model = keras.models.Sequential()

model.add(keras.layers.Dense(500, input_dim = dim ,activation = "relu"))

model.add(keras.layers.Dense(500,activation = "relu"))

model.add(keras.layers.Dense(100,activation = "relu", kernel_regularizer = l1(0.0001)))

#model.add(keras.layers.Dropout(0.6))

model.add(keras.layers.Dense(1))



##Polynomial

m1

opt = Adam(learning_rate=0.001)

model = keras.models.Sequential()

model.add(keras.layers.Dense(1, input_dim = dim ,activation = "linear"))





m2

opt = Adam(learning_rate=0.001)

model = keras.models.Sequential()

model.add(keras.layers.Dense(1, input_dim = dim ,activation = "linear"))



##

def overfit(dim):

    opt = Adam(learning_rate=0.0001)

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(500, input_dim = dim ,activation = "relu"))

    model.add(keras.layers.Dense(500,activation = "relu"))

    model.add(keras.layers.Dense(100,activation = "relu"))

    model.add(keras.layers.Dense(1))

    model.compile(loss = 'mse', optimizer = opt)

    return model



def mod_(dim):

    opt = Adam(learning_rate=0.0008)

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(500, input_dim = dim ,activation = "relu"))

    model.add(keras.layers.Dense(500,activation = "relu"))

    model.add(keras.layers.Dense(100,activation = "relu", kernel_regularizer = l1(0.0006)))

    #model.add(keras.layers.Dropout(0.05))

    model.add(keras.layers.Dense(1))

    model.compile(loss = 'mse', optimizer = opt)

    return model

'''
def overfit(dim):

    opt = Adam(learning_rate=0.0001)

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(500, input_dim = dim ,activation = "relu"))

    model.add(keras.layers.Dense(500,activation = "relu"))

    model.add(keras.layers.Dense(100,activation = "relu"))

    model.add(keras.layers.Dense(1))

    model.compile(loss = 'mse', optimizer = opt)

    return model



def mod_(dim):

    opt = Adam(learning_rate=0.0008)

    model = keras.models.Sequential()

    model.add(keras.layers.Dense(500, input_dim = dim ,activation = "relu"))

    model.add(keras.layers.Dense(500,activation = "relu"))

    model.add(keras.layers.Dense(100,activation = "relu", kernel_regularizer = l1(0.0006)))

    #model.add(keras.layers.Dropout(0.05))

    model.add(keras.layers.Dense(1))

    model.compile(loss = 'mse', optimizer = opt)

    return model
def app(

    noise,power,n_points,show_test,runmod1,runmod2, model_1_epochs, model_2_epochs,

    poly_features_m1,poly_features_m2

):

    f,ax = plt.subplots()

    

    polyx1 = PolynomialFeatures(poly_features_m1)

    polyx2 = PolynomialFeatures(poly_features_m2)

    

    x,y = poly(power,n_points)

    index = list(range(len(x)))

    np.random.shuffle(index)

    trainind = index[0:int(0.6*len(x))]

    testind = index[int(0.6*len(x)):]

    x_train,y_train = x[trainind],y[trainind]

    x_test,y_test = x[testind],y[testind]

    

    x_train = x_train + np.random.normal(0,noise,size = (len(x_train)))

    

    Temp = list(x_train) + list(x_test)

    

    plt.scatter(x_train,y_train, c = 'b')

    

    if show_test == True:

        plt.scatter(x_test,y_test, c= 'r')

        

    x_train = x_train.reshape(-1,1)

    x_test = x_test.reshape(-1,1)

    

    x_train1 = polyx1.fit_transform(x_train)[:,1:]

    x_test1 = polyx1.fit_transform(x_test)[:,1:]

    

    x_train2 = polyx2.fit_transform(x_train)[:,1:]

    x_test2 = polyx2.fit_transform(x_test)[:,1:]

    

    if runmod1 == True:

        mod = overfit(poly_features_m1)

        print ("Model1 retrained")

        mod.fit(x_train1,y_train,epochs = model_1_epochs, verbose = 0)

        x2 = np.linspace(min(Temp),max(Temp),1000)

        yhat = np.ndarray.flatten(mod.predict(polyx1.fit_transform(x2.reshape(-1,1))[:,1:]))

        yhat2 = np.ndarray.flatten(mod.predict(x_train1))

        yhat3 = np.ndarray.flatten(mod.predict(x_test1))

        plt.scatter(x2,yhat, c = 'purple', s = 1, alpha = 0.2)

        print ("MSE on train, model1 {}".format(round(sse(y_train,yhat2),7)))

        if show_test == True:

            print ("MSE on test, model1 {}".format(round(sse(y_test,yhat3),7)))

    else:

        pass

    

    if runmod2 == True:

        mod = mod_(poly_features_m2)

        print ("Model2 retrained")

        mod.fit(x_train2,y_train,epochs = model_2_epochs, verbose = 0)

        x2 = np.linspace(min(Temp),max(Temp),1000)

        yhat = np.ndarray.flatten(mod.predict(polyx2.fit_transform(x2.reshape(-1,1))[:,1:]))

        yhat2 = np.ndarray.flatten(mod.predict(x_train2))

        yhat3 = np.ndarray.flatten(mod.predict(x_test2))

        plt.scatter(x2,yhat, c = 'green', s = 1, alpha = 0.2)

        print ("SSE on train, model2 {}".format(round(sse(y_train,yhat2),7)))

        if show_test == True:

            print ("SSE on test, model2 {}".format(round(sse(y_test,yhat3),7)))

    else:

        pass

        

        

    

    

    plt.show()
style = {'description_width': 'initial'}

interactive_plot = interact(app,

            noise = BoundedFloatText(

    value=0,min=0,max=1,step=0.02,

    description='Level of Noise',

    orientation='horizontal',style=style

),

            power = Dropdown(

    options=[1,2,3,4,5,6,7,8],

    value=5,description='highest power',style=style

),

            n_points = BoundedIntText(

    value=20,min=0,max=1000,step=1,

    description='Total data points',

    orientation='horizontal',style=style

),

    show_test = Dropdown(

    options=[True,False],

    value=False,description='show test set',style=style

),

                            

    runmod1 = Dropdown(

    options=[True,False],

    value=False,description='show model 1',style=style

),

    model_1_epochs = BoundedIntText(

    value=3000,min=1,max=10000,step=1,

    description='Model 1 epochs',

    orientation='horizontal',style=style

),

                            

    runmod2 = Dropdown(

    options=[True,False],

    value=False,description='show model 2',style=style

),

                            

    model_2_epochs = BoundedIntText(

    value=3000,min=1,max=10000,step=1,

    description='Model 2 epochs',

    orientation='horizontal',style=style

),

    poly_features_m1 = Dropdown(

    options=list(range(1,21)),

    value=1,description='poly_features_m1',style=style

),

    poly_features_m2 = Dropdown(

    options=list(range(1,21)),

    value=1,description='poly_features_m2',style=style

),

                           )
reload_data = False                             #reload data from txt, otherwise from pickle



reload_model = True                            #retrain model, otherwise from pickle

model_dataset = "models"                        #name of the dataset contianing the trained models



grid_length = 30                               #grid lenght for interpolation



#values for recurent network

epochs = 100                                    #number of epochs

old_frames = 10                                 #number of frames taken to produce a new output

new_frames = 1                                  #number of new frames predicted

nb_dense_layers = 2                             #number of dense layers in the end

power_nb_nodes =    1.5                           #in each layer there are 2^(p*layer) nodes (multiplied by new_frames)



#values for pca

n_components=40                                #numbers of components (for arg and for angle)



#plot

epsilon = 0.05                                  #plot grid points grey, even if they have a small speed (in percentage of amax)


%matplotlib inline

import os

import numpy as np

import matplotlib.pyplot as plt



import copy

import pickle



from scipy.interpolate import griddata

from sklearn.decomposition import PCA



from keras.models import Sequential

from keras.layers import LSTM, Dense, Dropout, LeakyReLU, SimpleRNN, LSTM, GRU

from keras.models import model_from_json



%load_ext autoreload

%autoreload 2



# loads all the data from txt file, very time consuming, preferably just done once, afterwards use the pickled file

def load_data_all():

    #load the data

    print("start loading the data")

    data = np.loadtxt('/kaggle/input/avalanche/t_all.txt', dtype=np.float64, delimiter='\t', skiprows=1)

    print("data loaded from file")

    return data



# reshape the data in a 3dim array, save it afterwards as pickled file

def load_data_all_rearange(data):

    #reshape the data

    ta = np.ones((35862, 5, 501))

    for i in range(0, 501):

        ta[:,:,i] = data[35862*i:35862*(i+1), :]

        

    dt = ta[0,0,1]

    ta = np.delete(ta, 0, 1)

    

    #save it

    with open("t_all_formated.fichier", 'wb') as dumpfile:

        pickle.dump(ta, dumpfile)

    

    return ta, dt



def load_data_pickle():

    with open("/kaggle/input/avalanche/t_all_formated.fichier", 'rb') as dumpfile:

        ta = pickle.load(dumpfile)

    # supposed constant, could be read from original files

    dt = 0.0133333

    

    return ta, dt
def to_timeserie(data, old = old_frames, new = new_frames):

    #data contains (gridpoints)(arg_v, angle_v)(t)

    data = np.concatenate((data[:,0,:], data[:,1,:]), axis = 0)

    #resulting i (2*gridpoints)(t)

    l_ = (data.shape)[1]-old-new

    X, Y = np.zeros((l_,len(data),old)), np.zeros((l_, len(data),new))

    for i in range(l_):

        X[i,:] = data[:,i:(i+old)]

        Y[i,:] = data[:,(i+old):(i+old+new)]

    

    if new == 1:

        Y = np.reshape(Y, (Y.shape[0], Y.shape[1]))

    

    return X, Y
def points_to_grid(ta, m=0):

    methods=["nearest", "linear", "cubic"]

    xmin, xmax = np.amin(ta[:,0,:]), np.amax(ta[:,0,:])

    ymin, ymax = np.amin(ta[:,1,:]), np.amax(ta[:,1,:])

    

    s = ta.shape

    timesteps = s[2]

    

    grid_x, grid_y = np.mgrid[xmin:xmax:(grid_length*1j), ymin:ymax:(grid_length*1j)]

    

    gridvalues = np.zeros((grid_length, grid_length, 2, timesteps))

    gridvalues_f = np.zeros((grid_length*grid_length, 4, timesteps))

    for t in range(timesteps):

        points = ta[:,(0,1), t]

        v = ta[:, (2,3), t]

        

        arg = np.sqrt(np.power(v[:,0],2)+np.power(v[:,1],2))

        angle = np.arctan2(v[:,1], v[:,0])

        

        arg = np.reshape(arg, (len(arg),1))

        angle = np.reshape(angle, (len(angle),1))

        

        values = np.concatenate((arg, angle), axis = 1) 

        

        #artifical points with 0 speed in the middle of the gridpoints, to avoid speed outside avalanche

        points_0 = np.zeros((grid_length*grid_length,2))

        dx = grid_x[1][0]-grid_x[0][0]

        dy = grid_y[0][1]-grid_y[0][0]

        

        points_0[:,0] = grid_x.flatten()+dx/2;

        points_0[:,1] = grid_y.flatten()+dy/2;

        

        values_0 = np.zeros((grid_length*grid_length,2))

        

        #add them to original set -> method nearest neighbor will not work anymore

        points = np.concatenate((points, points_0), axis = 0)

        values = np.concatenate((values, values_0), axis = 0) 



        

        gridvalues[:,:,:, t] = griddata(points, values, (grid_x, grid_y), method=methods[m])

        

        gridvalues_f[:, 0, t] = grid_x.flatten()

        gridvalues_f[:, 1, t] = grid_y.flatten()

        gridvalues_f[:, (2,3), t] = np.reshape(gridvalues[:,:,:, t], (grid_length*grid_length, 2))

        

    return gridvalues_f

        

        
def grid_pca(ta_grid):

    pca_arg = PCA(n_components=n_components)

    pca_angle = PCA(n_components=n_components)

    

    #for argument of speed

    data_arg = ta_grid[:,2,:]

    data_arg = np.swapaxes(data_arg, 0, 1)

    

    pca_arg.fit(data_arg)

    data_pca_arg = pca_arg.transform(data_arg)

    print("singular values of pca_arg: {}".format(pca_arg.singular_values_))

    

    #for angle of speed

    data_angle = ta_grid[:,3,:]

    data_angle = np.swapaxes(data_angle, 0, 1)

    

    pca_angle.fit(data_angle)

    data_pca_angle = pca_angle.transform(data_angle)

    print("singular values of pca_angle: {}".format(pca_angle.singular_values_))

    

    # construct a subspace of the samples

    data_pca_arg_i = np.swapaxes(data_pca_arg, 0, 1)

    data_pca_angle_i = np.swapaxes(data_pca_angle, 0, 1)

    s = data_pca_arg_i.shape

    data_pca_arg_i = np.reshape(data_pca_arg_i, (s[0],1,s[1]))

    data_pca_angle_i = np.reshape(data_pca_angle_i, (s[0],1,s[1]))

    ta_red = np.concatenate((data_pca_arg_i,data_pca_angle_i), axis = 1)

    

    #take inverse of pca transform

    data_arg_r = pca_inverse(data_pca_arg, pca_arg)

    data_angle_r = pca_inverse(data_pca_angle, pca_angle)

    

    #reconstruct ta

    s = data_arg_r.shape

    data_arg_r = np.reshape(data_arg_r, (s[0],1,s[1]))

    data_angle_r = np.reshape(data_angle_r, (s[0],1,s[1]))

    ta_rec = np.concatenate((ta_grid[:,(0,1),:],data_arg_r,data_angle_r), axis = 1)

    

    return ta_rec, ta_red, pca_arg, pca_angle



def pca_inverse(data, pca):

    inv = pca.inverse_transform(data)  

    inv = np.swapaxes(inv, 0, 1)

    return inv
def generateRNN(trainX, trainY, mode = "simple", epochs=100):

    # for mode can be choosen between "RNN, LSTM and GRU"

    

    model = Sequential()

    

    nodes = int(trainX.shape[1]*np.power(1/power_nb_nodes,nb_dense_layers)*new_frames);

    

    if mode == "RNN":

        model.add(SimpleRNN(units=nodes, input_shape=(trainX.shape[1],old_frames), activation="relu"))

    elif mode == "LSTM":

        model.add(SimpleRNN(units=nodes, input_shape=(trainX.shape[1],old_frames), activation="relu"))

    elif mode == "GRU":

        model.add(SimpleRNN(units=nodes, input_shape=(trainX.shape[1],old_frames), activation="relu"))

    else:

        print("selected model not suported")

        exit()

    

    #bää, seems like we have to do upsampling...

    for l in range(nb_dense_layers):

        nodes = int(trainX.shape[1]*np.power(1/power_nb_nodes,nb_dense_layers-l-1)*new_frames);

        model.add(Dense(nodes, activation="relu")) 

    

    model.compile(loss='mean_squared_error', optimizer='rmsprop')

    model.summary()

    

    model.fit(trainX,trainY, epochs=epochs, batch_size=16, verbose=2)

    return model
def prediction(model, testX):

    predicted = model.predict(testX)

    pred_arg = predicted[:,0:n_components]

    pred_angle = predicted[:,n_components:2*n_components]



    """print("predict using inputdata")

    predicted = predicted[:,0]

    predicted = np.reshape(predicted, (len(predicted)))



    print(predicted.shape)

    #using the testX set would be kind of cheating, lets just use the predicted values to continue

    print("continue predicting, using prediction")

    for i in range(0, len(testX)-2*step):

        past_window = predicted[-step-1:-1]

        #print("window length is {}".format(len(past_window)))

        np_window = np.reshape(past_window, (1, 1, step))



        p = model.predict(np_window)

        p = np.reshape(p[0,0], (1))

        predicted=np.concatenate((predicted,p),axis=0)



    print("finished prediction l={}".format(predicted.shape))"""

    

    return pred_arg, pred_angle
def save_model(model, name="model"):

    # serialize model to JSON

    model_json = model.to_json()

    with open(name + ".json", "w") as json_file:

        json_file.write(model_json)

    # serialize weights to HDF5

    model.save_weights(name + ".h5")

    print("Saved model to disk")



def load_model(name="model", folder = "models"):

    # load json and create model

    json_file = open("/kaggle/input/"+folder+"/" + name + ".json", 'r')

    loaded_model_json = json_file.read()

    json_file.close()

    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model

    loaded_model.load_weights("/kaggle/input/"+folder+"/" + name + ".h5")

    print("Loaded model from disk")

    #not sure if compilation is necessary

    loaded_model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    

    return loaded_model
def plot_grid(title, **data):

    plot_scatter(title, grid=True, **data)



def plot_points(title, **data):

    plot_scatter(title, grid=False, **data)



def plot_scatter(title, grid, **kwargs):

    nb_keys = len(kwargs.keys())

    args = kwargs.values()

    

    #get min and max speed

    amin = float("inf")

    amax = float("-inf")

    for data in args:

        #get norm of speed

        if not grid:

            v = np.sqrt(np.power(data[:,2],2)+np.power(data[:,3],2))

        else:

            v = data[:,2]

            

        if np.amin(v) < amin:

            amin = np.amin(v)

        if np.amax(v) > amax:

            amax = np.amax(v)

    

    plt.subplots_adjust(hspace=0.6)

    if nb_keys > 3:

        fig, axs = plt.subplots(int(nb_keys/3)+1, 3, sharey = True, sharex = True, figsize=(8,5))

    else:

        fig, axs = plt.subplots(1, nb_keys, sharey = True, sharex = True, figsize=(8,5))



    fig.suptitle(title)

    

    for i, x in enumerate(kwargs.items()):

        [key, data] = x

        

        s=data.shape

        null = np.zeros((s[0],1))

        null = np.reshape(null, (s[0],1))

        

        if not grid:

            v = np.sqrt(np.power(data[:,2],2)+np.power(data[:,3],2))

        else:

            v = data[:,2]

        

        v = (v-amin)/(amax-amin)

        v = np.reshape(v, (s[0],1))

        

        # all particles at rest are shown green

        eps = epsilon

        

        r = v

        g = null

        b = (1-v)

        

        #make sure they are > 0

        r *= np.heaviside(v-eps, 0);

        g *= np.heaviside(v-eps, 0);

        b *= np.heaviside(v-eps, 0);

        

        grey = (1-np.heaviside(v-eps, 0))*0.9

        r += grey

        g += grey

        b += grey

        

        col = np.concatenate((r,g,b), axis = 1)

        

        

        if nb_keys == 1:

            axs.scatter(data[:,0], data[:,1], c=col, s=1)

            axs.set_xlabel("x")

            axs.set_ylabel("y")

            axs.set_title(key)

        elif nb_keys <= 3:

            axs[i].scatter(data[:,0], data[:,1], c=col, s=1)

            axs[i].set_xlabel("x")

            axs[i].set_ylabel("y")

            axs[i].set_title(key)

        else:

            r = i%3

            c = int(i/3)

            axs[c,r].scatter(data[:,0], data[:,1], c=col, s=1)

            axs[c,r].set_xlabel("x")

            axs[c,r].set_ylabel("y")

            axs[c,r].set_title(key)

    
if(reload_data):

    data = load_data_all()

    ta, dt = load_data_all_rearange(data)    

    

else:

    ta, dt = load_data_pickle()
ta_grid = points_to_grid(ta, m=0)
ta_rec, ta_red, pca_arg, pca_angle = grid_pca(ta_grid)
plot_points("original", t0=ta[:,:, 0], t50=ta[:,:, 50], t200=ta[:,:, 200], t300=ta[:,:, 300], t500=ta[:,:, 500])

plot_grid("to grid", t0=ta_grid[:,:, 0], t50=ta_grid[:,:, 50], t200=ta_grid[:,:, 200], t300=ta_grid[:,:, 300], t500=ta_grid[:,:, 500])

plot_grid("after pca", t0=ta_rec[:,:, 0], t50=ta_rec[:,:, 50], t200=ta_rec[:,:, 200], t300=ta_rec[:,:, 300], t500=ta_rec[:,:, 500])
#generate dataset for recurrent network



#using full space

if False:

    #its kind of cheating, but we train it in a first try with all timeframes

    trainX, trainY = to_timeserie(ta_grid[:,(2,3), :], old = old_frames, new = new_frames)



    #create several test sets, using different starting points



    l_train = 2*old_frames+1



    trainX0, trainY0 = to_timeserie(ta_grid[:,(2,3),0:l_train], old = old_frames, new = new_frames)

    trainX150, trainY150 = to_timeserie(ta_grid[:,(2,3),150:(150+l_train)], old = old_frames, new = new_frames)

    trainX300, trainY300 = to_timeserie(ta_grid[:,(2,3),300:(300+l_train)], old = old_frames, new = new_frames)



else:

    #its kind of cheating, but we train it in a first try with all timeframes

    trainX, trainY = to_timeserie(ta_red, old = old_frames, new = new_frames)



    #create several test sets, using different starting points



    l_train = 2*old_frames+1



    testX0, testY0 = to_timeserie(ta_red[:,:,0:l_train], old = old_frames, new = new_frames)

    testX150, testY150 = to_timeserie(ta_red[:,:,150:(150+l_train)], old = old_frames, new = new_frames)

    testX300, testY300 = to_timeserie(ta_red[:,:,300:(300+l_train)], old = old_frames, new = new_frames)
if reload_model:

    modelSimpleRNN = generateRNN(trainX, trainY,mode = "RNN", epochs=epochs)

    

    save_model(modelSimpleRNN, "simpleRNN_"+str(old_frames)+"_"+str(new_frames))

else:

    print("add dataset")

    modelSimpleRNN = load_model(name= "simpleRNN_"+str(old_frames)+"_"+str(new_frames), folder = model_dataset)
print(testX0.shape)

pred_arg, pred_angle = prediction(modelSimpleRNN, testX0)

print(pred_arg.shape)

print(pred_angle.shape)

s=pred_arg.shape

positions = ta_grid[:,(0,1),0:s[0]]

print(positions.shape)

inv_arg = pca_inverse(pred_arg, pca_arg)

inv_angle = pca_inverse(pred_angle, pca_angle)



s=inv_arg.shape

inv_arg = np.reshape(inv_arg, (s[0],1,s[1]))

inv_angle = np.reshape(inv_angle, (s[0],1,s[1]))

pred = np.concatenate((positions,inv_arg,inv_angle), axis = 1)

print(pred.shape)



                    
plot_grid("after reconstruction", t0=pred[:,:, 0], t1=pred[:,:, 1])
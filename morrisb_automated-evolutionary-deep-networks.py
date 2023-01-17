# To store data

import pandas as pd



# To do linear algebra

import numpy as np



# To create plots

import matplotlib.pyplot as plt



# To create nicer plots

import seaborn as sns



# To search in directories

import os



# To measure time

import time



# To flow tensors

from tensorflow.python import keras

from tensorflow.python.keras.models import Sequential, save_model, load_model

from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization



# To split the data

from sklearn.model_selection import train_test_split



# To augment the data

from keras.preprocessing.image import ImageDataGenerator



# To interpret strings

from ast import literal_eval
# Load the data

X_train = pd.read_csv('../input/mnist_train.csv')

X_test = pd.read_csv('../input/mnist_test.csv')

y_train = X_train.pop('label')





# Settings for the images

img_rows, img_cols = 28, 28

num_classes = 10





# Functions to modify the data

def prepareData(X_data):

    # Reshape and normalize the images

    num_images = X_data.shape[0]

    out_x = X_data.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x



def prepareLabel(y_data):

    # Transform labels to vectors

    out_y = keras.utils.to_categorical(y_data, num_classes)

    return out_y





# Prepare the images and labels for the network

X = prepareData(X_train.values)

y = prepareLabel(y_train.values)





# Check shapes

print('Image-Shape: {}'.format(X.shape))

print('Label-Shape: {}'.format(y.shape))
# Split the images into training and validation dataset

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)





#Data augmentation

datagen = ImageDataGenerator(rotation_range = 20,

                             zoom_range = 0.2,

                             width_shift_range = 0.2,

                             height_shift_range = 0.2,

                             shear_range = 20)

datagen.fit(X_train)
# Path to store the results of all individuals

evolution_path = 'evolution.csv'





# Check if there is an existing evolution to continue

if evolution_path in os.listdir():

    # Load training status of an existing evolution

    df = pd.read_csv(evolution_path)

    

else:

    # Create a new training status to start a new evolution

    df = pd.DataFrame(columns=['Epoch','Individual', 'Path', 'Duration', 'Layers', 'Settings', 'T_Loss', 'T_Score', 'V_Loss', 'V_Score', 'Error'])

    df.to_csv(evolution_path, index=False)





# Get current count values to continue the evolution with the correct indices

if df.empty:

    current_count = 0

    epoch_count = 0

    

else:

    current_count = df['Individual'].max()

    epoch_count = df['Epoch'].max()
# Functions to create layers with random parameters

# You can alter the settings for the possible parameters here



def createConv2D(new=True, settings=None, first=False):

    # (Re-)create a Conv2D-Layer

    if new:

        kernel = np.random.choice([1,3,5])

        filters = np.random.randint(1, 50)

    else:

        filters, kernel = settings

    if first:

        return Conv2D(filters, kernel_size=(kernel, kernel), activation='relu', input_shape=(img_rows, img_cols, 1))

    else:

        return Conv2D(filters, kernel_size=(kernel, kernel), activation='relu')



    

def createBatchNormalization():

    # (Re-)create a BatchNormalization-Layer

    return BatchNormalization(axis=1)





def createDropout(new=True, rate=None):

    # (Re-)create a Dropout-Layer

    if new:

        rate = np.random.random()

    return Dropout(rate=rate)





def createFlatten():

    # (Re-)create a Flatten-Layer

    return Flatten()





def createDense(new=True, dense=None, last=False):

    # (Re-)create a Dense-Layer

    if new:

        dense = np.random.randint(1, 128)

    if last:

        return Dense(dense, activation='softmax')

    else:

        return Dense(dense, activation='relu')





# List of layers to choose from for the evolution

# You can add new functions to create different layers here

layers = [createConv2D, createDense, createBatchNormalization, createDropout]
def createNewIndividual():

    # If a new evolution starts this function stocks up the first evolution-epoch with individual networks

    # Create model

    model = Sequential()

    

    # Start with a Convolution-Layer

    kernel_size = np.random.choice([1,3,5])

    model.add(Conv2D(np.random.choice([1,3,5]), kernel_size=(kernel_size, kernel_size), activation='relu', input_shape=(img_rows, img_cols, 1)))

    

    # Add flatten layer

    model.add(Flatten())

    

    # Add last Dense-Layer

    model.add(Dense(num_classes, activation='softmax'))

    

    # Compile the network

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model





def recreateEvolveIndividual(individual):

    # If an evolution status exists, this function recreates the choosen network and alters some settings randomly

    # Separate network-layers and layer-settings to recreate the network

    layers_string, settings = individual.values[0]

    layers_string = layers_string.split('_')

    settings = literal_eval(settings)

    

    # Random number to choose an action: delete/switch/insert layer

    random = np.random.random()

    # Random number to choose index for the action

    layer_index = np.random.randint(len(layers_string)-1)+1

    

    

    # Create model

    model = Sequential()

    

    # Iterate over all layers of the network and add them

    # Check if an alternation has to be implemented

    for i, (layer, setting) in enumerate(zip(layers_string, settings)):

        

        # Evolve randomly choosen layer

        if i==layer_index:

            

            # Delete layer (by not adding it)

            if random<0.25:

                # Pass the adding and continue with next layer

                pass

            

            # Switch layer

            elif random<0.75:

                # Instead of adding the layer, add a different layer

                model.add(np.random.choice(layers)())

                

            # Insert layer

            else:

                # Add a randomly choosen layer and add the normal layer afterwards

                model.add(np.random.choice(layers)())

                # Check for first/last layer since they have to be implemented differently (input-shape/activation-function)

                if i==0:

                    model.add(createConv2D(new=False, settings=setting, first=True))

                elif i+1==len(settings):

                    model.add(createDense(new=False, dense=setting, last=True))

                elif layer=='conv2d':

                    model.add(createConv2D(new=False, settings=setting))

                elif layer=='flatten':

                    model.add(createFlatten())

                elif layer=='dense':

                    model.add(createDense(new=False, dense=setting))

                elif layer=='batch':

                    model.add(createBatchNormalization())

                elif layer=='dropout':

                    model.add(createDropout(new=False, rate=setting))

                    

        # Add layer without evolution

        else:

            # Check for first/last layer since they have to be implemented differently (input-shape/activation-function)

            if i==0:

                model.add(createConv2D(new=False, settings=setting, first=True))

            elif i+1==len(settings):

                model.add(createDense(new=False, dense=setting, last=True))

            elif layer=='conv2d':

                model.add(createConv2D(new=False, settings=setting))

            elif layer=='flatten':

                model.add(createFlatten())

            elif layer=='dense':

                model.add(createDense(new=False, dense=setting))

            elif layer=='batch':

                model.add(createBatchNormalization())

            elif layer=='dropout':

                model.add(createDropout(new=False, rate=setting))

        

    # Compile the network

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model





def storeModel(df, model, epoch, j, t_loss, t_score, v_loss, v_score, end_time, error):

    # After training the network the results have to be saved to disc

    # Initialze some variables to store the results

    network = []

    network_settings = []

    

    # Iterate over all layers

    for layer in model.layers:

        # Add the name of the layer to recreate it

        name = layer.name.split('_')[0]

        network.append(name)



        # Add the settings of the layer to recreate the hyperparameters

        if name=='dense':

            settings = layer.units

        elif name=='conv2d':

            settings = [layer.filters, layer.kernel_size[0]]

        elif name=='dropout':

            settings = [layer.rate]

        else:

            settings = None

        network_settings.append(settings)

        

    # Combine the names of al layers

    network = '_'.join(network)

    

    # To save the model to disc

    # Check if training has been successfull

    # Save the network to disc to have the ability to reload it

    network_file = 'Saves/{}_{}.model'.format(epoch_count+epoch, current_count+j)

    #if not error:

        #save_model(model, network_file)



    # Append the training status and save it to disc

    df = df.append(pd.DataFrame([[epoch_count+epoch, current_count+j, network_file, end_time, network, network_settings, t_loss, t_score, v_loss, v_score, error]], columns=df.columns))

    df.to_csv(evolution_path, index=False)
# Number of individuals in the population

n = 10



# Number of epochs for the evolution

# Since tensorflow allocates some memory for each model trained it slows down dramatically after many networks

# Therefore I recommend to run the notebook several times locally

epochs = 5



# Iterate over all epochs

for epoch in range(1, epochs+1):

    # Iteration of individuals

    j = 0

    

    # Iterate over all individuals:

    for i in range(n):

        

        # Iterate until a valid individual has been (re-)created and trained

        iterate = True

        while iterate:



            # Load training history

            df = pd.read_csv(evolution_path)



            # Check if n individuals have been initialised for first generation

            if df[df['Error']==False].shape[0] < n:

                model = createNewIndividual()



            # First generation has been computed

            else:

                # Number of the current epoch

                current_epoch = df['Epoch'].max()



                # Choose next individual from weighted individuals

                evolvable_individual = df.sample(1, weights=df['V_Score']*df['Epoch'])[['Layers', 'Settings']]



                try:

                    # Evolve individual

                    model = recreateEvolveIndividual(evolvable_individual)

                except:

                    # Evolved model is not legit

                    continue



            

            # Start training process of the network

            try:

                batch_size = 100

                start_time = time.time()

                model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),

                                    epochs = 3, 

                                    validation_data = (X_val, y_val),

                                    verbose = 0, 

                                    steps_per_epoch = X_train.shape[0] // batch_size)

                end_time = time.time() - start_time



                # Retrieve training results

                t_loss = model.history.history['loss'][-1]

                t_score = model.history.history['acc'][-1]

                v_loss = model.history.history['val_loss'][-1]

                v_score = model.history.history['val_acc'][-1]



                print()

                iterate = False

                error = False

            

            # Set correct variables for untrainable network

            except:

                error = True

                t_loss = 0

                t_score = 0

                v_loss = 0

                v_score = 0

                end_time = 0

                pass

            j += 1

            

            # Store model and training results to disc

            try:

                storeModel(df, model, epoch, j, t_loss, t_score, v_loss, v_score, end_time, error)

            except:

                pass

            

            



# Since tensorflow slows the process down after several models you can use this function to restart and rerun your notebook locally

# An automatic stop for the rerunning has not been implemented

# Restarts the kernel and runs all cells

#from IPython.display import display_html



#def rerunKernel():

#    display_html("""<script>Jupyter.notebook.kernel.restart(); setTimeout( function(){ IPython.notebook.execute_all_cells(); }, 1000);</script>""",raw=True)

#rerunKernel()
# Reload the training results

df = pd.read_csv(evolution_path)



# Filter for the trained networks

df2 = df[df['Error']==False]



print('{} Networks have been tried'.format(df.shape[0]))

print('{} Networks have been trained'.format(df2.shape[0]))







# Plot the training duration for all networks

# You should see the increasing time for the training

# This occurs since tensorflo allocates memory for old models

df2['Duration'].plot(figsize=(10, 6), title='Training Duration For All Networks')

plt.xlabel('Network Individual [#]')

plt.ylabel('Training Duration [s]')

plt.show()







# Plot top n network architectures

n = 10



# Create order within the networks architectures

order = df2.groupby('Layers').max().sort_values('V_Score', ascending=False).index



plt.figure(figsize=(10,10))

sns.swarmplot(data=df2, x='V_Score', y='Layers', order=order[:n])

plt.title('Top {} Network Architectures - Best Score: {:.4f}'.format(n, df2['V_Score'].max()))

plt.grid()

plt.xlim(0.0, 1)

plt.show()







# Plot validation accuracy for the evolution proccess

n = 3



plt.figure(figsize=(12,6))

df3 = df2.groupby('Epoch').apply(lambda x: x.sort_values('V_Score', ascending=False).head(n))

sns.lineplot(data=df3, x='Epoch', y='V_Score')

plt.title('Validation Accuracy Of The Top {} Networks Of Each Epoch'.format(n))

plt.show()
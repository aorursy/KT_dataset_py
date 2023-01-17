import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse, os, cv2, pickle

from PIL import Image
from IPython.display import SVG
from tensorflow.keras.utils import plot_model, model_to_dot
from sklearn.utils import class_weight
from tqdm import tqdm

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Add, Input, Conv2D, Dense, MaxPooling2D, UpSampling2D, Activation)
from tensorflow.keras.optimizers import Adam, SGD, Adadelta
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback
from tensorflow.keras.initializers import *
from tensorflow.keras.regularizers import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def dense_autoencoder(input_shape):
    
    # Stage 1: Encoder Input
    encoder_input = Input(input_shape,name="encoder_input")
    
#     # Stage 2: Dense => ReLU
#     encoded = Dense(1024,name="encoder_1")(encoder_input)
#     encoded = Activation("relu",name="relu_encoder_1")(encoded)
    
    # Stage 2: Dense => ReLU
    encoded = Dense(512,name="encoder_2")(encoder_input)
    encoded = Activation("relu",name="relu_encoder_2")(encoded)
    
    # Stage 3: Dense => ReLU
    encoded = Dense(256,name="encoder_3")(encoded)
    encoded = Activation("relu",name="relu_encoder_3")(encoded)
    
    # Stage 4: Dense => ReLU
    encoded = Dense(64,name="encoder_4")(encoded)
    encoded = Activation("relu",name="relu_encoder_4")(encoded)
    
    # Encoder model
    encoder = Model(inputs=encoder_input,outputs=encoded)
    
    # Stage 5: Decoder Input
    decoder_input = Input(shape=(64,),name="decoder_input")
    
    # Stage 6: Dense => ReLU
    decoded = Dense(256,name="decoder_1")(decoder_input)
    decoded = Activation("relu",name="relu_decoder_1")(decoded)
    
    # Stage 7: Dense => ReLU
    decoded = Dense(512,name="decoder_2")(decoded)
    decoded = Activation("relu",name="relu_decoder_2")(decoded)
    
#     # Stage 9: Dense => ReLU
#     decoded = Dense(1024,name="decoder_3")(decoded)
#     decoded = Activation("relu",name="relu_decoder_3")(decoded)
    
    # Stage 8: Dense => ReLU
    decoded = Dense(input_shape[0],name="decoder_4")(decoded)
    decoded = Activation("relu",name="relu_decoder_4")(decoded)
    
    # Decoder Model
    decoder = Model(inputs=decoder_input,outputs=decoded)
    
    # Creating autoencoder model
    # # Autoencoder input
    autoencoder_input = Input(input_shape)
    encoded_autoencoder = encoder(autoencoder_input)
    decoded_autoencoder = decoder(encoded_autoencoder)
    
    autoencoder = Model(inputs=autoencoder_input,outputs=decoded_autoencoder)
    
    return autoencoder, encoder, decoder
    pass
autoencoder, encoder, decoder = dense_autoencoder(input_shape=(11520,))
# autoencoder_plot = os.path.join("output/autoencoder_plot.png")
plot_model(autoencoder,to_file="autoencoder_plot.png",show_shapes=True,show_layer_names=True)
SVG(model_to_dot(autoencoder).create(prog='dot',format='svg'))
autoencoder.summary()
# encoder_plot = os.path.join(config.BASE_CSV_PATH,"encoder_plot.png")
plot_model(encoder,to_file = "encoder_plot.png",show_shapes = True,show_layer_names = True)
SVG(model_to_dot(encoder).create(prog='dot',format='svg'))
encoder.summary()
# decoder_plot = os.path.join(config.BASE_CSV_PATH,"decoder_plot.png")
plot_model(encoder,to_file="decoder_plot.png",show_shapes=True,show_layer_names=True)
SVG(model_to_dot(decoder).create(prog='dot',format='svg'))
decoder.summary()
opt = Adadelta(lr=1e-3)
autoencoder.compile(optimizer=opt,loss="mse")
logs = TensorBoard("logs")
# checkpoint = ModelCheckpoint("model_weights.h5",monitor="val_accuracy",
#                             verbose=1,save_best_only=True,mode='max')
def feature_generator(inputPath):
    
    df = pd.read_csv(inputPath,header=None)
    
    filenames = df.loc[:,0]
    return np.array(filenames)
    pass
def data_generator(inputPath, batchSize, mode="train"):
    
    df = pd.read_csv(inputPath,header=None)
    num_samples = df.shape[0]
    
    while True:
        
        for offset in range(0, num_samples, batchSize):
            batchSamplesIdx = df.index[offset:offset+batchSize]
            
            X, y = [], []
            
            for i in batchSamplesIdx:
            
                feature = df.loc[i,2].split(" ")
                feature = np.array(feature, dtype=np.float32)
                X.append(feature)
                y.append(feature)
                pass
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            yield X, y
            pass
        pass
    pass
le = pickle.load(open("../input/image-search-engine/ImageSearch/dataset/label_map.pkl","rb"))
le
numClasses = len(le.keys())
numClasses
trainPath = "../input/testdata/train.csv"
valPath = "../input/testdata/validation.csv"
%%time
train_filenames = feature_generator(trainPath)
val_filenames = feature_generator(valPath)

train_filenames.shape, val_filenames.shape
train_generator = data_generator(trainPath, 2)
val_generator = data_generator(valPath, 2)
numEpochs = 100
batchSize = 19

dense_history = autoencoder.fit(train_generator,
                               steps_per_epoch = 1832//batchSize,
                               epochs = numEpochs,
                               verbose = 1,
                               validation_data = val_generator,
                               validation_steps = 532//batchSize,
                               callbacks=[logs])
def show_final_history(history):
    
    plt.style.use("ggplot")

    plt.plot(history.history['loss'],label='Train Loss')
    plt.plot(history.history['val_loss'],label='Validation Loss')
    
#     ax[0].legend(loc='upper right')
#     ax[1].legend(loc='lower right')
    plt.show();
    pass
show_final_history(dense_history)
def feature_generator(inputPath):
    
    df = pd.read_csv(inputPath,header=None)
    numSamples = df.shape[0]
    
    features = np.zeros(11520)
    
    for i in tqdm(range(0, numSamples, 1)):
        
        feature = np.array(df.loc[i,2].split(" "), dtype=np.float32)
        features = np.vstack((features, feature))
        pass
    
    return features
    pass
X_train = feature_generator(trainPath)
X_train.shape
X_train = np.delete(X_train, 0, 0)
X_train.shape
val_train = feature_generator(valPath)
val_train.shape
val_train = np.delete(val_train, 0, 0)
val_train.shape
%%time
X_train_encoded = encoder.predict(X_train)
X_val_encoded = encoder.predict(val_train)

X_train_encoded.shape, X_val_encoded.shape
filenames = np.concatenate((train_filenames,val_filenames),axis=0)
encoded_values = np.concatenate((X_train_encoded,X_val_encoded), axis=0)

filenames.shape, encoded_values.shape
output_file = open("encoded_values_train_val.csv","w")
for (filename, vec) in zip(filenames, encoded_values):
    vec = ",".join([str(v) for v in vec])
    output_file.write("{},{}\n".format(filename, vec))
    pass

df_encoded = pd.read_csv("./encoded_values_train_val.csv",header=None)
df_encoded.head()
encoder.save("encoder_model.h5")
decoder.save("decoder_model.h5")
autoencoder.save("autoencoder_model.h5")
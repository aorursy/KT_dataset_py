import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Activation, Input, Conv2D, Add, Flatten, AveragePooling2D, Dense, Dropout, BatchNormalization
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import h5py
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
sample = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
train.describe()
X = train.drop("label", axis = 1).values.reshape(-1, 28, 28, 1)
y = train["label"].values

test = test.values.reshape(-1, 28, 28, 1)

def normalize(data):
    return (data/255)

X = normalize(X)
test = normalize(test) 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex = True, sharey = True)
axs = [ax1, ax2, ax3, ax4]
for i in range(len(axs)):
    axs[i].imshow(X[i][:, :, 0])
def blocks(x, strides, filters):
    y = x
    
    x = Conv2D(filters, kernel_size = 3, padding = "same", strides = strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters, kernel_size = 3, padding = "same", strides = (1, 1))(x)
    x = BatchNormalization()(x)
    
    
    if strides[0] > 1:
        y = Conv2D(filters, kernel_size = 3, padding = "same", strides = strides)(y)
        y = BatchNormalization()(y)
        
    x = Add()([x, y])
    return x
def residual_net(input_shape, num_classes, filters, blks):
    inputs = Input(input_shape)
    x = Conv2D(filters, kernel_size = 7, padding = "same", strides = (1, 1))(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    # Adding residual blocks
    for block in blks:
        x = blocks(x, strides = (2, 2), filters = filters)
        for i in range(block - 1):
            x = blocks(x, strides = (1, 1), filters = filters)
        filters *= 2
        
    x = AveragePooling2D(4)(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(int(filters/4), activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    
    # Instantiate model
    model = Model(inputs=inputs, outputs=outputs)
    return model       
        
model = residual_net(
    input_shape=X[0].shape, 
    num_classes=np.unique(y).shape[-1], 
    filters=64, 
    blks=[2,2,2]
)
resnet_architecture = plot_model(model, show_shapes=True, show_layer_names=False)
resnet_architecture.width = 600
resnet_architecture
def train(epochs, batch_size, i):
    model = residual_net(
    input_shape=X[0].shape, 
    num_classes=np.unique(y).shape[-1], 
    filters=64, 
    blks=[3,3,3])
    
    model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    
    checkpoint = ModelCheckpoint(
    filepath=f'resnet-{i}.hdf5',
    monitor='loss',
    save_best_only=True
    )
    def scheduler(epoch):
        return 0.001 * 0.8**epoch
    schedule = LearningRateScheduler(scheduler)
    callbacks = [checkpoint, schedule]
    
    train_datagen = ImageDataGenerator(  
        rotation_range=10,  
        zoom_range=0.1, 
        width_shift_range=0.1, 
        height_shift_range=0.1)
    
    train_datagen.fit(X)

    # Fit model
    history = model.fit_generator(
        train_datagen.flow(X, y, batch_size=batch_size),
        epochs=epochs, 
        verbose=2, 
        callbacks=callbacks)
    
    return history, model
    
    
    
models = []
all_history = []
for i in range(5):
    print("#"*20)
    print(f"Model number {i + 1}")
    print("#"*20)
    
    history, model = train(
        epochs=12,
        batch_size=128, i = i)
    
    models.append(model)
    all_history.append(history)
preds = []
for model in models:
    preds.append(model.predict(test))
    
f_pred = np.zeros(preds[0].shape)
for p in preds:
    f_pred = f_pred + p
    
preds_mean = (f_pred)/5
predictions = np.argmax(preds_mean, axis=1)
result = pd.Series(predictions, name = "Label")
sample["Label"] = result
sample.set_index("ImageId", inplace = True)
sample.to_csv("submission_resnet.csv")

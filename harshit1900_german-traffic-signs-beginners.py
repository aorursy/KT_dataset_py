!nvidia-smi
import pandas as pd
import numpy as np
df_train = pd.read_csv('../input/gtsrb-german-traffic-sign/Train.csv')
df_train['Path'] = df_train['Path'].str.lower()
df_train['ClassId'] = df_train['ClassId'].apply(str)
df_train.head()
df_train.tail()
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(dtype='int8', sparse=False) #Sparse matrix: Most of the elements are zero. int8: Byte (-128 to 127)
y_train = ohe.fit_transform(df_train['ClassId'].values.reshape(-1,1)) #Reshape:To make sure the new shape must be compatible with the original shape
import keras
from tqdm import tqdm
from keras.preprocessing import image
train_img = []                          # Creating a list
for i in tqdm(range(df_train.shape[0])):
    img = image.load_img('../input/gtsrb-german-traffic-sign/' + df_train['Path'][i], target_size = (64, 64, 3)) #Loading the images and giving the dimensions to the image
    img = image.img_to_array(img)  #For converting images to arrays
    img = img/255 #Normalizing the images by bringing them into same scale by dividing the RGB values by 255
    train_img.append(img)   #Storing the preprocessed images in the list
X = np.array(train_img)
X.shape #Check the shape of training images
import matplotlib 
from matplotlib import pyplot as plt
plt.imshow(X[3909])     #Checking a particular image
df_test = pd.read_csv('../input/gtsrb-german-traffic-sign/Test.csv')
df_test['Path'] = df_test['Path'].str.lower()
df_test['ClassId'] = df_test['ClassId'].apply(str)
df_test.head()
df_test.tail()
test_img = []
for i in tqdm(range(df_test.shape[0])):
    img = image.load_img('../input/gtsrb-german-traffic-sign/' + df_test['Path'][i], target_size = (64, 64, 3))
    img = image.img_to_array(img)
    img = img/255
    test_img.append(img)
y = np.array(test_img)
y.shape
plt.imshow(y[1000])
from keras.models import Sequential  #Helps to create models in a layer by layer architecture
from keras.layers import Dense   #Layers which are connected to each other    
from keras.layers import Conv2D #Convolution 2D: Class of Neural networks that specializes in processing data that has a grid like topology such as an image. Creates a kernel which is further connected wth the input layer
from keras.layers import AveragePooling2D #For reducing the spatial size of the images progressiviely. Average value is taken
from keras.layers import Flatten #For converting the data into a 1-dimensional array for inputting it to the next layer.
from keras.layers import Dropout #For ignoring some of the neurons during the training phase
from keras.layers import BatchNormalization #Normalizes the output of previous activation layer. Inceases stability in the network
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape = (64, 64, 3), activation = 'relu', name = 'first')) #No. of filters: 64: Used for determining No. of kernels to convolve with the input volume.
model.add(AveragePooling2D(pool_size = (2, 2))) 
model.add(BatchNormalization())
model.add(Dropout(0.30)) 

model.add(Conv2D(64, (3, 3), activation = 'relu'))  #Activation: For deciding whether the neuron is to be activated or not by calculating weighted sum.
model.add(AveragePooling2D(pool_size = (2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.30))

model.add(Flatten())

model.add(Dense(128, activation='relu')) #Units: No. of neurons/cells in a layer. ReLU is preffered beacuse of its sparsity and reduced likelihood of vanishing gradient problem 
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(43, activation='softmax', name = 'last')) #Output layer. 
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy']) #CCE: For Multiclass problems
model.summary()
from sklearn.model_selection import train_test_split #For splitting data into training and test sets
train_df, df_validate = train_test_split(df_train, test_size = 0.30, random_state = 42) #Test_Size: By default 0.25, random_state: for generating random integers when the code is run
train_df = train_df.reset_index(drop = True) #reset_index: sets a list of integer ranging from 0 to length of data as index.
df_test = df_test.reset_index(drop = True)
df_validate = df_validate.reset_index(drop = True)
train_df['ClassId'].value_counts()
df_validate['ClassId'].value_counts()
train_df['ClassId'].value_counts().plot.bar()
plt.show()
df_validate['ClassId'].value_counts().plot.bar()
plt.show()
train_total = train_df.shape[0] #Gives first component of dimensions 'train_df'
validate_total = df_validate.shape[0]
batch_size = 64
from keras.preprocessing.image import ImageDataGenerator #Importing library for Generation of images
train_data_generator = ImageDataGenerator(
        rotation_range = 15,
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
    )
train_gen = train_data_generator.flow_from_dataframe(
        dataframe = train_df,
        directory = '../input/gtsrb-german-traffic-sign/',
        x_col = 'Path',
        y_col = 'ClassId',
        target_size = (64, 64),
        batch_size = batch_size,
        class_mode = 'categorical',
    )
validate_data_generator = ImageDataGenerator(
        rotation_range = 15,
        rescale = 1./255, #Scaling the value from 0 to 1 from 0-255
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
)
validate_gen = validate_data_generator.flow_from_dataframe(
        dataframe = df_validate,
        directory = '../input/gtsrb-german-traffic-sign/',
        x_col = 'Path',
        y_col = 'ClassId',
        target_size = (64, 64),
        batch_size = batch_size,
        class_mode = 'categorical'
 )
history = model.fit_generator(
        train_gen,  
        epochs = 50,   #One pass over the entire data
        steps_per_epoch = 150, #One update of the parameters
        validation_data = validate_gen,
        validation_steps = 50
  )

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
!nvidia-smi
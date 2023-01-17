# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# import libaries 



import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

import cv2 as cv

import random



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



import tensorflow.keras 

from tensorflow.keras.models import Sequential, load_model, Model

from tensorflow.keras import layers 

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

from tensorflow.keras.metrics import Precision, Recall, AUC

from tensorflow.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

# import the data

train_df = pd.read_csv('../input/digit-recognizer/train.csv')

test_df = pd.read_csv('../input/digit-recognizer/test.csv')

train_df.head()
num_instances = train_df.groupby('label').size()



plt.figure(figsize = (10,5))

plt.bar(np.unique(train_df.label),num_instances)

plt.title('Number of labels within the training set', fontweight = 'bold')

plt.xlabel('labels')

plt.ylabel('instances')
%%time

samples, columns = train_df.shape



# empty tensors 

X = np.zeros((samples,28,28,1))

y_true = np.zeros((samples,1))



for sample in tqdm(range(samples)):

    X[sample,:,:,:] = train_df.iloc[sample,1:columns].values.reshape(28,28,1).astype('float32') # convert vectors into 2D tensors with (28,28,1)

    y_true[sample,0] = train_df.iloc[sample,0] # read the the corresponding output labels
values = train_df.label

# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(values)



print('The original output labels', values)

# binary encode

onehot_encoder = OneHotEncoder(sparse=False)



integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded) # corresponding loss function is the "categorical cross entropy" and the neural network output should be a layer with 9 neurons

samples, classes = onehot_encoded.shape

print("Number of vectors:", samples, "\nNumber of neurons / length of vector:", classes)
y = onehot_encoded # corresponding ground truth vector for X
%%time

# normalize the input features

def standard_norm(img):

    return (img - np.mean(img))/np.std(img)



# empty tensor 

norm_X = np.zeros((samples,28,28,1))

for sample in tqdm(range(samples)):

    norm_X[sample,:,:,:] = standard_norm(X[sample,:,:,:]).reshape(28,28,1) 

    
def METRICS():

    metrics = ['accuracy', 

              Precision(name='precision'), 

              Recall(name='recall'),

              AUC(name='AUC')]

    return metrics





model = Sequential()

model.add(layers.Input(shape=(28, 28, 1))) 

model.add(layers.Conv2D(32, (3,3), padding = 'same', activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Conv2D(64, (3,3), padding = 'same', activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.MaxPooling2D(pool_size = (2, 2)))

model.add(layers.Conv2D(128, (3,3), padding = 'same', activation='relu'))

model.add(layers.BatchNormalization())





model.add(layers.GlobalAveragePooling2D()) 

model.add(layers.Dense(classes,activation='softmax', name = 'output_layer'))

model.compile(Adam(lr = 0.00100005134), metrics= METRICS(), loss = 'categorical_crossentropy') 

model.summary()
# functions to help split our data and train our model ...



def split_data(X,Y):

    return train_test_split(X, Y, test_size=0.2, random_state=42)



def train_model(model, X, Y, epochs, bs):

    X_train, X_val, y_train, y_val = split_data(X,Y)

    

    STEP_SIZE_TRAIN = X_train.shape[0]//bs + 1

    STEP_SIZE_VAL = X_val.shape[0]//bs + 1

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    

    train_history = model.fit(X_train, y_train, 

                             steps_per_epoch = STEP_SIZE_TRAIN,

                             validation_data = (X_val,y_val),

                             validation_steps = STEP_SIZE_VAL, 

                            epochs = epochs, shuffle = True,

                             )

    return train_history, model

# train model .. 

epochs, bs = 20, 32 # choosen hyperparameters

train_hist, final_model = train_model(model, norm_X, y, epochs, bs)
%%time



samples, columns = test_df.shape

X_test = np.zeros((samples,28,28,1)) # empty tensor

X_norm_test = np.zeros((samples,28,28,1))

for sample in tqdm(range(samples)):

    X_test[sample,:,:,:] = test_df.iloc[sample,:].values.reshape(28,28,1).astype('float32') # convert vector into 2D tensor

    X_norm_test[sample,:,:,:] = standard_norm(X_test[sample,:,:].reshape(28,28,1))
"""

    Implementing class activation maps for architectures with Global Average Pooling 2D before the final dense layer 

"""

class MNIST_CAM:

    

    def __init__(self, img):

        self.resize_width, self.resize_height, _ = img.shape    

    

    # zero-center normalization 

    def standard_norm(self, img):

        return ((img - np.mean(img))/np.std(img))

    

    # final layer should be (7,7,2048)

    def feature_model(self, model):  

        return Model(inputs = model.layers[0].input, outputs = model.layers[-3].output)

    

    # final weight tensor before classification layer is 3*2048

    def weight_tensor(self, model):

        final_outputs = model.layers[-1]

        return final_outputs.get_weights()[0]

    

    # output prediction class of the image of interest

    def predict_class(self, model, X):

        prob_vec = model.predict(X)

        return np.argmax(prob_vec[0])

        

    # generate class activation maps (CAMs)    

    def generate_CAM(self, model, img):

        norm_img = self.standard_norm(img)

        Fmap_model = self.feature_model(model)

        Wtensor = self.weight_tensor(model)

        feature_map = Fmap_model.predict(norm_img.reshape(1,28,28,1))

        label = self.predict_class(model, norm_img.reshape(1,28,28,1))

        CAM = feature_map.dot(Wtensor[:,label])[0,:,:]

        return cv.resize(CAM, 

                         (self.resize_width, self.resize_height),

                         interpolation = cv.INTER_CUBIC), label

    

    # generate probability vector 

    def generate_probvec(self, model, img):

        X = self.standard_norm(img)

        prob_vec = model.predict(X.reshape(1,28,28,1))

        return prob_vec
# example image 

img = X_test[102,:,:,:]

CAM_generator = MNIST_CAM(img)

plt.imshow(img.reshape(28,28), cmap='gray')

activation_map, label = CAM_generator.generate_CAM(final_model, img)

plt.imshow(activation_map,'jet', alpha = 0.3)

plt.title("Predicted Class: " + str(label))

plt.show()
# Here is an interactive loop that asks if you want to continue to generate random input digit images 

#through the CAM_generator ...





# Generate and plot class activation map along with the original image ... 

while True:

    sample = random.randint(0, len(X_test))

    img = X[sample,:,:,:] 

    CAM_generator = MNIST_CAM(img)

    plt.imshow(img.reshape(28,28), cmap='gray')

    activation_map, label = CAM_generator.generate_CAM(final_model, img) # generate activation map and output label

    plt.imshow(activation_map,'jet', alpha = 0.3)

    plt.title("Predicted Class: " + str(label))

    plt.show()

    request = input("Next Image? (y/n)")

    if request and request[0] == 'n':

        break
final_model.predict(X_norm_test[0,:,:,:].reshape(1,28,28,1))
y_test,test_Ids = np.zeros((samples,1)), np.zeros((samples,1))





for sample in tqdm(range(samples)):

    y_test[sample,0] = np.argmax(final_model.predict(X_norm_test[sample,:,:,:].reshape(1,28,28,1)))

    test_Ids[sample,0] = int(sample+1)          
label_df, pred_df = pd.DataFrame(test_Ids), pd.DataFrame(y_test)

sub_df = pd.concat([label_df, pred_df], axis = 1)

sub_df.iloc[:,:] = sub_df.iloc[:,:].astype('int')

sub_df.columns = ['ImageId', 'Label']

sub_df.head()
sub_df.to_csv('sample_submission.csv', index=False)
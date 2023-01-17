%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(123)
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
import itertools
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
import itertools
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical 
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

sns.set(color_codes = True)

#Global Variables
styles=[':','-.','--','-',':','-.','--','-',':','-.','--','-']
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
# Create a dictionary with image_id, image_path as key value pairs. 
base_skin_dir = os.path.join('..', 'input')

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(base_skin_dir,'*','*','*.jpg'))}

# This dictionary is useful for displaying more human-friendly labels later on
lesion_type_dict = {
                    'nv': 'Melanocytic nevi',
                    'mel': 'Melanoma',
                    'bkl': 'Benign keratosis-like lesions ',
                    'bcc': 'Basal cell carcinoma',
                    'akiec': 'Actinic keratoses',
                    'vasc': 'Vascular lesions',
                    'df': 'Dermatofibroma'
                    }
skin_df = pd.read_csv(os.path.join(base_skin_dir,'skin-cancer-mnist-ham10000','HAM10000_metadata.csv'))

skin_df['path'] = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type'] = skin_df['dx'].map(lesion_type_dict.get) 
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes

skin_df.head()
skin_df.isnull().sum()
skin_df.age.describe()
skin_df["age"].fillna(skin_df.age.median(), inplace = True)
sns.set(font_scale = 1.25)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 2, figsize=(20, 25))
sns.despine(left=True)

# Age distribution
sns.kdeplot(skin_df["age"], legend = False, shade = True, ax=axes[0, 0])
axes[0,0].set_xlabel("Age", fontsize=17)
axes[0,0].set_title("Age: Distribution Plot", fontsize=20)

# Gender distribution
sns.countplot(x = "sex", data = skin_df, ax=axes[0, 1])
axes[0,1].set_xlabel("Gender", fontsize=17)
axes[0,1].set_title("Gender: Distribution Plot", fontsize=20)
axes[0,1].set_ylabel("Count", fontsize=17)

# Diagnosis Test type distribution
sns.countplot(x = "dx_type", data = skin_df, ax=axes[1, 0])
axes[1,0].set_xlabel("Diagnosis Test Type", fontsize=17)
axes[1,0].set_ylabel("Count", fontsize=17)
axes[1,0].set_title("Diagnosis Test Type: Distribution Plot", fontsize=20)

# Lesion type distribution
sns.countplot(x = "cell_type", data = skin_df, ax=axes[1, 1])
axes[1,1].set_xlabel("Lesion Type", fontsize=17)
axes[1,1].set_ylabel("Count", fontsize=17)
axes[1,1].set_title("Lesion Type: Distribution Plot", fontsize=20)

## Lesion type distribution
sns.countplot(x = "localization", data = skin_df, ax=axes[2, 0])
axes[2,0].set_xlabel("Localization Area", fontsize=17)
axes[2,0].set_ylabel("Count", fontsize=17)
axes[2,0].set_title("Localization Area: Distribution Plot", fontsize=20)

c = 0
for ax in f.axes:
    c+=1
    if c<=3:
        continue
    plt.sca(ax)
    plt.xticks(rotation=90)
    
plt.subplots_adjust(top=0.95)
f.suptitle('Univariate Distributions', fontsize=25)
f.delaxes(axes[2,1]) 
sns.set(font_scale = 1.25)

# Set up the matplotlib figure
f, axes = plt.subplots(1, 2, figsize=(20, 15))
sns.despine(left=True)

# Gender distribution
sns.boxplot(x = "sex", y = "age", data = skin_df, ax=axes[0])
axes[0].set_xlabel("Gender", fontsize=17)
axes[0].set_ylabel("Age", fontsize=17)
axes[0].set_title("Gender V. Age Boxplot", fontsize=20)

# Lesion distribution
sns.boxplot(x = "cell_type", y = "age", data = skin_df, ax=axes[1])
axes[1].set_xlabel("Lesion Type", fontsize=17)
axes[1].set_ylabel("Age", fontsize=17)
axes[1].set_title("Lesion Type V. Age Boxplot", fontsize=20)

c = 0
for ax in f.axes:
    c+=1
    if c<=1:
        continue
    plt.sca(ax)
    plt.xticks(rotation=90)
%%time
skin_df['image'] = skin_df['path'].map(lambda x: np.asarray(Image.open(x).resize((100,75))))
skin_df.head()
n_samples = 5
fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
for n_axs, (type_name, type_rows) in zip(m_axs, skin_df.sort_values(['cell_type']).groupby('cell_type')):
    n_axs[0].set_title(type_name)
    for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=1234).iterrows()):
        c_ax.imshow(c_row['image'])
        c_ax.axis('off')
# fig.savefig('category_samples.png', dpi=300)

# Checking the image size distribution
skin_df['image'].map(lambda x: x.shape).value_counts()
features = skin_df["image"]
target = skin_df["cell_type_idx"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.1, random_state = 999)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.165, random_state = 999)
print("Shape of entire dataset: {}".format(str(features.shape)))
print("Shape of train data: {}".format(str(X_train.shape)))
print("Shape of test data: {}".format(str(X_test.shape)))
print("Shape of val data: {}".format(str(X_val.shape)))
l = ["train","val","test"]

for x in l:
    globals()["X_{}".format(x)] = np.asarray(globals()["X_{}".format(x)].tolist())
    globals()["X_{}_mean".format(x)] = np.mean(globals()["X_{}".format(x)])
    globals()["X_{}_std".format(x)] = np.std(globals()["X_{}".format(x)])
    globals()["X_{}".format(x)] = (globals()["X_{}".format(x)]-globals()["X_{}_mean".format(x)])/globals()["X_{}_std".format(x)]
# Perform one-hot encoding on the labels
y_train = to_categorical(y_train, num_classes = 7)
y_test = to_categorical(y_test, num_classes = 7)
y_val = to_categorical(y_val, num_classes = 7)
# Reshape image in 3 dimensions (height = 75px, width = 100px , canal = 3)
X_train = X_train.reshape(X_train.shape[0], *(75, 100, 3))
X_test = X_test.reshape(X_test.shape[0], *(75, 100, 3))
X_val = X_val.reshape(X_val.shape[0], *(75, 100, 3))
input_shape = (75, 100, 3)
num_classes = 7
# With data augmentation to prevent overfitting 

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
nets = 3
input_shape = (75, 100, 3)
num_classes = 7
model = [0]*nets

for i in range(nets):
    model[i] = Sequential()
    model[i].add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
    model[i].add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    if i>0:
        model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(MaxPool2D(pool_size = (2,2)))
    if i>1:
        model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu', input_shape = input_shape))
        model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Flatten())
    model[i].add(Dense(256, activation = 'relu'))
    model[i].add(Dense(num_classes, activation = 'softmax'))
    model[i].compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])        
history = [0]*nets
names = ['Model 1.1','Model 1.2','Model 1.3']
epochs = 20 
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                        epochs = epochs, validation_data = (X_val,y_val),
                                        verbose = 0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
          epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))
# Plot Model Performance
plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.72,0.79])
plt.show()
nets = 3
model = [0]*nets
for i in zip(range(nets),[32,64,128]):
    model[i[0]] = Sequential()
    model[i[0]].add(Conv2D(filters = i[1], kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(Conv2D(filters = i[1], kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(MaxPool2D(pool_size = (2,2)))
    model[i[0]].add(Conv2D(filters = i[1]*2, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(Conv2D(filters = i[1]*2, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(MaxPool2D(pool_size = (2,2)))
    model[i[0]].add(Conv2D(filters = i[1]*4, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(Conv2D(filters = i[1]*4, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i[0]].add(MaxPool2D(pool_size = (2,2)))
    model[i[0]].add(Flatten())
    model[i[0]].add(Dense(256, activation = 'relu'))
    model[i[0]].add(Dense(num_classes,activation = 'softmax'))
    model[i[0]].compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = [0]*nets
names = ['Model 2.1','Model 2.2','Model 2.3']
epochs = 20 
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                        epochs = epochs, validation_data = (X_val,y_val),
                                        verbose = 0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
          epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))
# Plot Model Performance
plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.73,.78])
# axes.set_xlim([0,20])
plt.show()
nets = 2
model = [0]*nets

for i in range(2):
    model[i] = Sequential()
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
    model[i].add(MaxPool2D(pool_size = (2,2)))
    model[i].add(Flatten())
    if i == 0:
        model[i].add(Dense(512, activation = 'relu'))
    elif i == 1:
        model[i].add(Dense(1024, activation = 'relu'))
    model[i].add(Dense(num_classes,activation = 'softmax'))
    model[i].compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = [0]*nets
names = ['512N','1024N']
epochs = 20 
batch_size = 50
for i in range(nets):
    history[i] = model[i].fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                        epochs = epochs, validation_data = (X_val,y_val),
                                        verbose = 0, steps_per_epoch=X_train.shape[0] // batch_size,
                                        callbacks=[learning_rate_reduction])
    print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(names[i],
          epochs,max(history[i].history['accuracy']),max(history[i].history['val_accuracy']) ))
# Plot Model Performance
nets = 2
names = ['512N','1024N']

plt.figure(figsize=(15,5))
for i in range(nets):
    plt.plot(history[i].history['val_accuracy'],linestyle=styles[i])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(names, loc='upper left')
axes = plt.gca()
axes.set_ylim([0.72,.78])
plt.show()
model = Sequential()
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(Conv2D(filters = 256, kernel_size = (3,3), padding = 'same', activation = 'relu',input_shape = input_shape))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(BatchNormalization())
model.add(Dense(num_classes,activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=6)
mod_chckpt = ModelCheckpoint(filepath='model_v1.h5', monitor='val_loss', save_best_only=True)
history = [0]
epochs = 25
batch_size = 50

history[0] = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
                                epochs = epochs, validation_data = (X_val,y_val),
                                verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size,
                                callbacks=[learning_rate_reduction, early_stop, mod_chckpt])
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
loss_v, accuracy_v = model.evaluate(X_val, y_val, verbose=1)
print("Validation: accuracy = %f  ;  loss_v = %f" % (accuracy_v, loss_v))
print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
df_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")
df.head()
# You can see the Total number of rows and columns here
df.info()
df.describe()
count = df['label'].value_counts()
print(count)
# Also, just for verification
count.sum()
fig1 = plt.figure(figsize=(5,3),dpi=100)
axes = fig1.add_axes([1,1,1,1])
axes.set_ylim([0, 1300])
axes.set_xlabel('classes')
axes.set_ylabel('No. of Examples available')
axes.set_xlim([0,24])
for i in range(24):
    axes.axvline(i)
axes.bar(count.index,count.values,color='purple',ls='--') 
# I like the palette 'magma' too much. You can go with others like 'coolwarm','hsl' or 'husl'
sns.countplot(x=df['label'], palette='magma')
image_labels = df_test['label']
del df['label']
fig, axes = plt.subplots(3,4,figsize=(10,10),dpi=150)
k = 1
for i in range(3):
    for j in range(4):
        axes[i,j].imshow(df.values[k].reshape(28,28))
        axes[i,j].set_title("image "+str(k))
        k+=1
        plt.tight_layout()
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                # print("Ignoring first line")
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
        #print(labels)
    return images, labels

path_sign_mnist_train = f"{getcwd()}/../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)
training_images = np.expand_dims(training_images, axis=-1)
testing_images = np.expand_dims(testing_images, axis=-1)

# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=10,
                                   width_shift_range=0.2,
                                   height_shift_range=0.1,
                                   zoom_range=0.2)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
# Define the model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
ankitz = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.75, min_lr=0.00001)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (4, 4), activation='relu', input_shape=(28, 28, 1),padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',bias_regularizer=regularizers.l2(1e-4),),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (2, 2), activation='relu', padding='same',bias_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1,1), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(384, activation=tf.nn.relu, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(25, activation=tf.nn.softmax,kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.l2(1e-4))])
model.summary()
# Compile Model. 
model.compile(optimizer=tf.optimizers.Adam(),
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# Train the Model
history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=128),
                              epochs = 30,
                              validation_data=validation_datagen.flow(testing_images, testing_labels),
                             callbacks = [ankitz])

print("\n\nThe accuracy for the model is: "+str(model.evaluate(testing_images, testing_labels, verbose=1)[1]*100)+"%")
# Plot the chart for accuracy and loss on both training and validation
%matplotlib inline
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
predictions = model.predict_classes(testing_images)
print(predictions[:15])
print(image_labels.values[:15])
from sklearn.metrics import classification_report
classes_report = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(image_labels, predictions, target_names = classes_report))
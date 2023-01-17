import tensorflow as tf

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.layers import Dropout

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,

shear_range = 0.2,

zoom_range = 0.2,

horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('data/train',

target_size = (64, 64),

batch_size = 32,

class_mode = 'categorical')



test_set = test_datagen.flow_from_directory('data/public_test',

target_size = (64, 64),

batch_size = 32,

class_mode = 'categorical')
# Initialising the CNN

classifier = Sequential()

# Step 1 - Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second axpooling

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size= (2, 2)))
# Step 3 - Flattening

classifier.add(Flatten())



# Step 4 - Full connection

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(units = 8, activation = 'softmax'))

# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Train model

classifier.fit_generator(training_set,

steps_per_epoch = 6589,

epochs = 5,

validation_data = test_set,

validation_steps = 20)
classifier.save('my_model.h5')
new_model = tf.keras.models.load_model('my_model.h5')

new_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
dirs = os.listdir('data/data_private')

file = open("solve.csv", "a")



for files in dirs:

    file_name = "data/data_private/" + files

    img = cv2.imread(file_name)

    img = cv2.resize(img,(64,64))

    img = np.reshape(img,[1,64,64,3])

    classes = new_model.predict_classes(img)

    x = int(classes)

    file.write(files)

    file.write(",")

    file.write(str(x))

    file.write("\n")



file.close()
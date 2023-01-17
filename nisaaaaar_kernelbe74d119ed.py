# Importing the Keras libraries and packages

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
# Initialising the CNN

classifier = Sequential()
# Convolution

classifier.add(Conv2D(32, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
# Pooling

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a third convolutional layer

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a fourth convolutional layer

classifier.add(Conv2D(64, (3, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Flattening

classifier.add(Flatten())
# Full connection

classifier.add(Dense(units = 512, activation = 'relu'))

classifier.add(Dense(units = 256, activation = 'relu'))

classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 75, activation = 'softmax'))
# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Fitting the CNN to the images



from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

        rescale=1./255,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        '../input/dataset_2/dataset/train',

        target_size=(256, 256),

        batch_size=10,

        class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

        '../input/dataset_2/dataset/test',

        target_size=(256, 256),

        batch_size=10,

        class_mode='categorical')



classifier.fit_generator(

        train_generator,

        steps_per_epoch=7850,

        epochs=4,

        validation_data=validation_generator,

        validation_steps=1200)
import numpy as np

from keras.preprocessing import image

test_img=image.load_img('../input/dataset_2/dataset/predict/16.jpg',target_size=(256,256))

test_img = np.expand_dims(test_img, axis = 0)

result = classifier.predict(test_img)

print(result)

train_generator.class_indices

if result[0][0] == 1:

    prediction = 'Motorbike'

elif result[0][1] == 1:

    prediction = 'airplane'

elif result[0][2] == 1:

    prediction = 'backpack'

elif result[0][3] == 1:

    prediction = 'binocular'

elif result[0][4] == 1:

    prediction = 'bowling-pin'

elif result[0][5] == 1:

    prediction = 'brain'

elif result[0][6] == 1:

    prediction = 'buddha'

elif result[0][7] == 1:

    prediction = 'cake'

elif result[0][8] == 1:

    prediction = 'calculator'

elif result[0][9]== 1:

    prediction = 'camera'

elif result[0][10]== 1:

    prediction = 'cannon'

elif result[0][11]== 1:

    prediction = 'car-tire'

elif result[0][12]== 1:

    prediction = 'cd'

elif result[0][13]== 1:

    prediction = 'ceiling_fan'

elif result[0][14]== 1:

    prediction = 'cereal-box'

elif result[0][15]== 1:

    prediction = 'chair'

elif result[0][16]== 1:

    prediction = 'chess-board'

elif result[0][17]== 1:

    prediction = 'chopsticks'

elif result[0][18]== 1:

    prediction = 'coffee=mug'

elif result[0][19]== 1:

    prediction = 'coin'

elif result[0][20]== 1:

    prediction = 'computer-keyboard'

elif result[0][21]== 1:

    prediction = 'computer-monitor'

elif result[0][22]== 1:

    prediction = 'computer-mouse'

elif result[0][23]== 1:

    prediction = 'cowboy-hat'

elif result[0][24]== 1:

    prediction = 'cup'

elif result[0][25]== 1:

    prediction = 'desk-globe'

elif result[0][26]== 1:

    prediction = 'diamond-ring'

elif result[0][27]== 1:

    prediction = 'dice'

elif result[0][28]== 1:

    prediction = 'dog'

elif result[0][29]== 1:

    prediction = 'doorknob'

elif result[0][30]== 1:

    prediction = 'drinking-straw'

elif result[0][31]== 1:

    prediction = 'dumb-bell'

elif result[0][32]== 1:

    prediction = 'eyeglass'

elif result[0][33]== 1:

    prediction = 'fire-extinguisher'

elif result[0][34]== 1:

    prediction = 'fire-truck'

elif result[0][35]== 1:

    prediction = 'flashlight'

elif result[0][36]== 1:

    prediction = 'hamburger'

elif result[0][37]== 1:

    prediction = 'head-phones'

elif result[0][38]== 1:

    prediction = 'helicopter'

elif result[0][39]== 1:

    prediction = 'hot-dog'

elif result[0][40]== 1:

    prediction = 'ice-cream-cone'

elif result[0][41]== 1:

    prediction = 'knife'

elif result[0][42]== 1:

    prediction = 'laptop'

elif result[0][43]== 1:

    prediction = 'lightbulb'

elif result[0][44]== 1:

    prediction = 'mountain-bike'

elif result[0][45]== 1:

    prediction = 'mug'

elif result[0][46]== 1:

    prediction = 'mushroom'

elif result[0][47]== 1:

    prediction = 'necktie'

elif result[0][48]== 1:

    prediction = 'paperclip'

elif result[0][49]== 1:

    prediction = 'people'

elif result[0][50]== 1:

    prediction = 'photocopier'

elif result[0][51]== 1:

    prediction = 'playing-card'

elif result[0][52]== 1:

    prediction = 'pyramid'

elif result[0][53]== 1:

    prediction = 'revolver'

elif result[0][54]== 1:

    prediction = 'rifle'

elif result[0][55]== 1:

    prediction = 'scissors'

elif result[0][56]== 1:

    prediction = 'screwdriver'

elif result[0][57]== 1:

    prediction = 'sneaker'

elif result[0][58]== 1:

    prediction = 'soccer-ball'

elif result[0][59]== 1:

    prediction = 'socks'

elif result[0][60]== 1:

    prediction = 'spoon'

elif result[0][61]== 1:

    prediction = 'stapler'

elif result[0][62]== 1:

    prediction = 'stop_sign'

elif result[0][63]== 1:

    prediction = 'syringe'

elif result[0][64]== 1:

    prediction = 't-shirt'

elif result[0][65]== 1:

    prediction = 'teapot'

elif result[0][66]== 1:

    prediction = 'teddy-bear'

elif result[0][67]== 1:

    prediction = 'tennis-ball'

elif result[0][68]== 1:

    prediction = 'traffic-light'

elif result[0][69]== 1:

    prediction = 'tricycle'

elif result[0][70]== 1:

    prediction = 'umbrella'

elif result[0][71]== 1:

    prediction = 'video-projector'

elif result[0][72]== 1:

    prediction = 'watch'

elif result[0][73]== 1:

    prediction = 'wheelchair'

elif result[0][74]== 1:

    prediction = 'yo-yo'

else:

    prediction = 'could not identify'

print(prediction)
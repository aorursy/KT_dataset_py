from IPython.display import display



# Global variables

image_size = (32, 32)
from keras.preprocessing.image import load_img

from os import listdir

from os.path import join

from pandas import read_csv



# Take a look at the dataset

train_labels = read_csv('../input/train.csv')

for image_name in listdir('../input/train/train')[:10]:

    image = load_img(join('../input/train/train', image_name), target_size=image_size)

    display(train_labels[train_labels['id'] == image_name]['has_cactus'].item())

    display(image)
from keras.applications.vgg19 import VGG19, preprocess_input

from keras.preprocessing.image import img_to_array, load_img

from keras.utils import to_categorical

from numpy import array

from os import listdir

from os.path import join

from pandas import read_csv

from tqdm import tqdm_notebook 



def extract_features(label_path, set_path):

    # Load all images and the corresponding labels

    images = []

    labels = []



    # Create a pre-trained model

    model = VGG19(include_top=False, input_shape=(image_size[0], image_size[1], 3))



    train_labels = read_csv(label_path)

    for image_name in tqdm_notebook (listdir(set_path)):

        image = load_img(join(set_path, image_name), target_size=image_size)

        images.append(img_to_array(image))

        label = train_labels[train_labels['id'] == image_name]['has_cactus'].item()

        labels.append(label)



    training_images = preprocess_input(array(images))

    training_labels = array(labels)



    # Use the pre-trained network to extract features

    features = model.predict(training_images)

    

    return features, training_labels
from joblib import dump

from os import listdir



features, training_labels = extract_features('../input/train.csv', '../input/train/train')



# Save the features and labels to files

dump(features, 'features.dat')

dump(training_labels, 'labels.dat')



display(listdir('.'))
from joblib import load

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten 

from matplotlib.pyplot import legend, plot, show, title, xlabel, ylabel

from pathlib import Path



# Load the features and labels

x_train = load('features.dat')

y_train = load('labels.dat')



# Build the model

model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()



# Compile and fit

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

fit_model = model.fit(x_train, y_train, epochs=50, shuffle=True, validation_split=0.05)



# summarize history for accuracy

plot(fit_model.history['acc'])

plot(fit_model.history['val_acc'])

title('model accuracy')

ylabel('accuracy')

xlabel('epoch')

legend(['train', 'test'], loc='upper left')

show()



# summarize history for loss

plot(fit_model.history['loss'])

plot(fit_model.history['val_loss'])

title('model loss')

ylabel('loss')

xlabel('epoch')

legend(['train', 'test'], loc='upper left')

show()



# Save the trained network

Path('model_structure.json').write_text(model.to_json())

model.save_weights('model_weights.h5')



display(listdir('.'))
from csv import writer

from keras.applications.vgg19 import preprocess_input

from keras.models import model_from_json

from keras.preprocessing.image import img_to_array

from numpy import array

from pathlib import Path

from tqdm import tqdm_notebook 



# Load our trained model

model_structure = Path('model_structure.json').read_text()

model = model_from_json(model_structure)

model.load_weights('model_weights.h5')



images = []

for image_name in tqdm_notebook(listdir('../input/test/test')):

    image = load_img(join('../input/test/test', image_name), target_size=image_size)

    images.append(img_to_array(image))

    

images_to_predict = preprocess_input(array(images))



feature_extractor = VGG19(include_top=False, input_shape=(image_size[0], image_size[1], 3))

features = feature_extractor.predict(images_to_predict)

predictions = model.predict(features)



display(predictions)



with open('submission.csv', 'w+') as submissionCsvFile:

    csvWriter = writer(submissionCsvFile, lineterminator='\n')

    csvWriter.writerow(['id', 'has_cactus'])

    

    for index, image_name in enumerate(listdir('../input/test/test')):        

        csvWriter.writerow([image_name, predictions[index][0]])
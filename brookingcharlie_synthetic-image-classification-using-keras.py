INPUT='../input'

SYNIMG=f'{INPUT}/synimg'
#!kaggle competitions download -p $INPUT -c synthetic-image-classification
#!unzip -u -d $SYNIMG $INPUT/synimg.zip
!ls $INPUT
!ls $SYNIMG/synimg
import pandas as pd



styles = pd.read_csv(f'{SYNIMG}/synimg/styles.txt', names=['style_name'])
styles
!ls $SYNIMG/synimg/train
import pandas as pd



train = pd.read_csv(f'{SYNIMG}/synimg/train/data.csv')
train.head()
train.shape
train.groupby('style_name')[['style_name']].count()
from IPython.display import Image



for style in styles['style_name']:

    display(style)

    for filepath in train[train['style_name'] == style]['filepath'][0:3]:

        display(Image(f'{SYNIMG}/{filepath}'))
!ls $SYNIMG/synimg/test
import pandas as pd



test = pd.read_csv(f'{SYNIMG}/synimg/test/data_nostyle.csv')
test.head()
test.shape
from IPython.display import Image



for filepath in test['filepath'][0:3]:

    display(Image(f'{SYNIMG}/{filepath}'))
import keras



image = keras.preprocessing.image.load_img(f'{SYNIMG}/synimg/test/A/test-A-9000000.jpg')
image.height, image.width
image.getbands()
image_shape = (image.height, image.width, len(image.getbands()))
image_shape
import sklearn.preprocessing



label_encoder = sklearn.preprocessing.LabelBinarizer()

label_encoder.fit(styles['style_name'])
import numpy as np



display(label_encoder.classes_)

display(label_encoder.transform(['HongKong', 'Zurich', 'Syndey', 'Zurich']))

display(label_encoder.inverse_transform(np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])))
import numpy as np

import keras

import sklearn.model_selection



def load_image(filepath):

    return np.asarray(keras.preprocessing.image.load_img(f'{SYNIMG}/{filepath}')) / 255.0



def load_images(filepaths):

    return np.asarray([load_image(filepath) for filepath in filepaths])



def load_data(df):

    images = load_images(df['filepath'])

    labels = label_encoder.transform(df['style_name'])

    return sklearn.model_selection.train_test_split(images, labels, test_size=0.25)
import unittest



class TestLoadImages(unittest.TestCase):

    def test_load_image(self):

        result = load_image('synimg/test/A/test-A-9000000.jpg')

        self.assertTrue(isinstance(result, np.ndarray))

        self.assertEqual(result.dtype, 'float64')

        self.assertEqual(result.shape, image_shape)

        self.assertTrue((result >= 0.0).all() and (result <= 1.0).all())

    def test_load_images(self):

        result = load_images(['synimg/test/A/test-A-9000000.jpg', 'synimg/test/B/test-B-9000001.jpg'])

        self.assertEqual(result.shape, (2, *image_shape))



class TestLoadData(unittest.TestCase):

    def test_load_data(self):

        df = pd.DataFrame({

            'style_name': [

                'Luanda',

                'Luanda',

                'Brisbane',

                'Brisbane'

            ],

            'filepath': [

                'synimg/train/Luanda/train-Luanda-1000000.jpg',

                'synimg/train/Luanda/train-Luanda-1000001.jpg',

                'synimg/train/Brisbane/train-Brisbane-1090000.jpg',

                'synimg/train/Brisbane/train-Brisbane-1090001.jpg'

            ]

        })

        X_train, X_test, y_train, y_test = load_data(df)

        self.assertEqual((X_train.shape, y_train.shape), ((3, *image_shape), (3, len(styles))))

        self.assertEqual((X_test.shape, y_test.shape), ((1, *image_shape), (1, len(styles))))



unittest.main(argv=[''], exit=False)
train_images, test_images, train_labels, test_labels = load_data(train)
from keras import layers, models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=image_shape))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(len(styles), activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20)
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc
test_predictions = model.predict(test_images)
import pandas as pd



test_predictions_df = pd.DataFrame({

    'expected': label_encoder.inverse_transform(test_labels),

    'actual': label_encoder.inverse_transform(test_predictions)

})

pd.crosstab(test_predictions_df['expected'], test_predictions_df['expected'] == test_predictions_df['actual'], normalize='index')
real_test_images = load_images(test['filepath'])

predictions = model.predict(real_test_images)

prediction_labels = label_encoder.inverse_transform(predictions)

prediction_labels
!head $INPUT/sample_submission.csv
submission = test[['id']].assign(style_name = prediction_labels)

submission.head()
submission.to_csv('submission.csv', index=False)
!head submission.csv
#!kaggle competitions submit -c synthetic-image-classification -f submission.csv -m ''
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
TRAIN_FILE = '../input/fashion-mnist_train.csv'

TEST_FILE = '../input/fashion-mnist_test.csv'
import tensorflow as tf



tf.logging.set_verbosity(tf.logging.ERROR)



print('TensorFlow version: ', tf.__version__)
MODEL_DIR = './fashion-model'
feature_columns = [tf.feature_column.numeric_column('pixels', shape=[28,28])]
classifier = tf.estimator.LinearClassifier(

    feature_columns=feature_columns,

    n_classes=10,

    model_dir=MODEL_DIR

)
def generate_labelled_input_fn(csv_files, batch_size):

    def input_fn():

        file_queue = tf.train.string_input_producer(csv_files)

        reader = tf.TextLineReader(skip_header_lines=1)

        _, rows = reader.read_up_to(file_queue, num_records=100*batch_size)

        expanded_rows = tf.expand_dims(rows, axis=-1)

        

        shuffled_rows = tf.train.shuffle_batch(

            [expanded_rows],

            batch_size=batch_size,

            capacity=20*batch_size,

            min_after_dequeue=5*batch_size,

            enqueue_many=True

        )



        record_defaults = [[0] for _ in range(28*28+1)]



        columns = tf.decode_csv(shuffled_rows, record_defaults=record_defaults)



        labels = columns[0]



        pixels = tf.concat(columns[1:], axis=1)



        return {'pixels': pixels}, labels

    

    return input_fn
BATCH_SIZE = 40
TRAIN_STEPS = 2000
classifier.train(

    input_fn=generate_labelled_input_fn([TRAIN_FILE], BATCH_SIZE),

    steps=TRAIN_STEPS

)
classifier.evaluate(

    input_fn=generate_labelled_input_fn([TEST_FILE], BATCH_SIZE),

    steps=100

)
import csv

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd
CLASSES = {

    '0': 'T-shirt/top',

    '1': 'Trouser',

    '2': 'Pullover',

    '3': 'Dress',

    '4': 'Coat',

    '5': 'Sandal',

    '6': 'Shirt',

    '7': 'Sneaker',

    '8': 'Bag',

    '9': 'Ankle boot'

}
test_data = pd.read_csv(TEST_FILE)
sample_row = test_data.sample()
sample = list(sample_row.iloc[0])

label = sample[0]

pixels = sample[1:]
image_array = np.asarray(pixels, dtype=np.float32).reshape((28, 28))


plt.imshow(image_array, cmap='gray')
def generate_prediction_input_fn(image_arrays):

    def input_fn():

        queue = tf.train.input_producer(

            tf.constant(np.asarray(image_arrays)),

            num_epochs=1

        )

        

        image = queue.dequeue()

        return {'pixels': [image]}

    

    return input_fn
predictions = classifier.predict(

    generate_prediction_input_fn([image_array]),

    predict_keys=['probabilities', 'classes']

)
prediction = next(predictions)
print('Prediction output: {}'.format(prediction))
print('Actual label: {} - {}'.format(label, CLASSES[str(label)]))

predicted_class = prediction['classes'][0].decode('utf-8')

probability = prediction['probabilities'][int(predicted_class)]

print('Predicted class: {} - {} with probability {}'.format(

    predicted_class,

    CLASSES[predicted_class],

    probability

))
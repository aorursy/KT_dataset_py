import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
import tensorflow as tf
from bees import bees_utils as utils
utils.data_folder = '../input/data'
np.random.seed(117)
tf.set_random_seed(117)
img_width = 100
img_height = 100
img_channels = 3
bees, bees_test_for_evaluation = utils.read_data()
bees.head()
bees_test_for_evaluation.head()
utils.value_counts(bees, 'subspecies')
utils.plot_images(bees, 'location', [0, 18, 24, 38, 45])
train_bees, val_bees, test_bees = utils.split(bees)
train_X, val_X, test_X, train_y, val_y, test_y = utils.load_images_and_target(train_bees, 
                                                                              val_bees, 
                                                                              test_bees,
                                                                              'subspecies',
                                                                              img_width, 
                                                                              img_height,
                                                                              img_channels)
def class_weights(df, class_name) :
    # Hint: usar
    # http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    # y = df[class_name]
    return np.ones(utils.categories[class_name].shape[0])

optimizer = 'adam'
loss = 'categorical_crossentropy'
model1 = Sequential()
model1.add(Flatten(input_shape =(img_height, img_width, img_channels)))
model1.add(Dense(train_y.columns.size, activation = 'softmax'))
model1.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
rotation_range = 15      # rotación aleatoria en grados entre 0 a rotation_range
zoom_range = 0.1         # zoom aleatorio
width_shift_range = 0.1  # desplazamiento horizontal aleatorio (fracción del total)
height_shift_range = 0.1 # desplazamiento vertical aleatorio (fracción del total)
horizontal_flip = True   # transposición horizontal
vertical_flip = True     # transposición horizontal
batch_size = 10
epochs = 5
steps_per_epoch = 10
patience = 10
class_weights = class_weights(bees, 'subspecies')
class_weights
training1, model1 = utils.train(model1,
                train_X,
                train_y, 
                batch_size = batch_size,
                epochs = epochs,
                validation_data_X = val_X, 
                validation_data_y = val_y,
                steps_per_epoch = steps_per_epoch,
                rotation_range = rotation_range,
                zoom_range = zoom_range, 
                width_shift_range = width_shift_range,
                height_shift_range = height_shift_range,
                horizontal_flip = horizontal_flip,  
                vertical_flip = vertical_flip,
                patience = patience,
                class_weights = class_weights
                               )
utils.eval_model(training1, model1, test_X, test_y, 'subspecies')
df_test_1 = utils.load_test_and_generate_prediction_file(model1, class_weights, 'subspecies', img_width, img_height, img_channels, 'test_1.csv')
df_test_1.head(10)

from keras.models import Sequential

from keras.layers import Dense

from keras.utils.vis_utils import plot_model

model = Sequential()

model.add(Dense(units=4, activation='relu', input_shape=(4,)))

model.add(Dense(units=10, activation='relu'))

model.add(Dense(units=3, activation='sigmoid'))

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
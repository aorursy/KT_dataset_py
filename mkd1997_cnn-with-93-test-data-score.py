#all imports

import numpy as np

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

from keras.wrappers.scikit_learn import KerasClassifier
#data preprocessing

img_rows, img_cols = 28, 28

num_classes = 10



def prep_data(raw):

    y = raw[:, 0]

    out_y = keras.utils.to_categorical(y, num_classes)

    

    x = raw[:,1:]

    num_images = raw.shape[0]

    out_x = x.reshape(num_images, img_rows, img_cols, 1)

    out_x = out_x / 255

    return out_x, out_y



#train data

fashion_train_file = "../input/fashion-mnist_train.csv"

fashion_train_data = np.loadtxt(fashion_train_file, skiprows=1, delimiter=',')

x_train, y_train = prep_data(fashion_train_data)



#test data

fashion_test_file = "../input/fashion-mnist_test.csv"

fashion_test_data = np.loadtxt(fashion_test_file, skiprows=1, delimiter=',')

x_test, y_test = prep_data(fashion_test_data)
#model creation : CNN

def create_model():

    fashion_model = Sequential()



    fashion_model.add(Conv2D(12, kernel_size=(3,3), activation="relu", input_shape=(img_rows, img_cols, 1)))

    fashion_model.add(Conv2D(20, kernel_size=(3,3), activation="relu"))

    fashion_model.add(MaxPooling2D(pool_size=(2,2)))

    fashion_model.add(Dropout(0.5))

    fashion_model.add(Flatten())

    fashion_model.add(Dense(100, activation="relu"))

    fashion_model.add(Dense(num_classes, activation="softmax"))



    fashion_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return fashion_model



# model = KerasClassifier(build_fn=create_model, verbose=0)        FOR hyperparamater optimizaton

fashion_model = create_model()

fashion_model.fit(x_train, y_train, batch_size=120, epochs=17, validation_split=0.2)
# hyperparameter optimization

# batch_size = list(np.arange(80,160,20))

# epochs = list(np.arange(1,20,4))



# param_grid = dict(batch_size=batch_size, epochs=epochs)

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3, verbose=0)

# grid_result = grid.fit(x_train, y_train)

# print(grid_result.best_score_)

# print(grid_result.best_params_)



# This has been commented to save execution time. 

# The best batch sze an no. of epochs are 120 and 17 respectively.

# Uncomment the above snippet and run the cell to see for  yourself
#model testing : CNN

fashion_model.evaluate(x_test, y_test, batch_size=120)

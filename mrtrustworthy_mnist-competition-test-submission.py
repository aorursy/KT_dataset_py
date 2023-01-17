import numpy as np 

import pandas as pd 

from tensorflow import keras



from subprocess import check_output

print('Available data:', check_output(["ls", "../input"]).decode("utf8"))

with open('../input/train.csv') as infile:

    first_line = [l for l in infile][1]

    print(first_line)
def get_data():

    contents = pd.read_csv(f'../input/train.csv', sep=',', header=0)

    # in the training data, the first column are the labels

    data = contents.iloc[:,1:]

    labels = contents.iloc[:,0]

    # convert them into a one-hot encoding. 

    # Note: don't be confused by the method name, this does actually a one-hot encoding and not a dummie one

    labels_onehot = pd.get_dummies(labels)

    images = []

    labels = []

    for i in range(len(data)):

        # we convert the numbers into floats, reshape it to (28,28,1) and then 

        # "normalize" it simply by bringing all possible values into a range from 0.0 to 1.0

        image = data.iloc[i].as_matrix().astype(np.float32).reshape((28,28,1)) / 255.

        images.append(image)

        labels.append(labels_onehot.iloc[i].astype(np.float32))

    return np.asarray(images), np.asarray(labels)



def get_test_data():

    contents = pd.read_csv(f'../input/test.csv', sep=',', header=0)

    # in the test data, there are no labels in the dataset

    data = contents.iloc[:,:]

    images = []

    for i in range(len(data)):

        image = data.iloc[i].as_matrix().astype(np.float32).reshape((28,28,1)) / 255.

        images.append(image)

    return np.asarray(images)

    

    

data, labels = get_data()

test_data = get_test_data()

print('Shape of the data is', x.shape, 'and shape of the labels is', y.shape)

def get_model():

    # input shape is the same as described above. Note that we don't include the sample size here!

    X = X_input = keras.layers.Input((28, 28, 1))

    

    # Some convolutional layers that basically act as feature extractor

    X = keras.layers.Conv2D(8, (3, 3), padding='same')(X)

    X = keras.layers.Activation('relu')(X)

    X = keras.layers.BatchNormalization(axis=-1)(X)  # Normalize input based on the channels



    X = keras.layers.Conv2D(16, (3, 3), padding='same')(X)

    X = keras.layers.Activation('relu')(X)

    X = keras.layers.MaxPooling2D(pool_size=(2,2))(X)  # reduced output to 14x14

    

    X = keras.layers.Conv2D(32, (3, 3), padding='same')(X)

    X = keras.layers.Activation('relu')(X)

    

    X = keras.layers.Conv2D(64, (3, 3), padding='same')(X)

    X = keras.layers.Activation('relu')(X)

    X = keras.layers.MaxPooling2D(pool_size=(2,2))(X)  # reduces output to 7x7

    

    # Some fully connected layers that basically make the decision 

    # regarding class assignment based on the extracted features

    X = keras.layers.Flatten()(X)

    

    X = keras.layers.Dense(1024)(X)

    X = keras.layers.Dropout(0.6)(X)

    X = keras.layers.Dense(1024)(X)

    X = keras.layers.Dropout(0.5)(X)

    X = keras.layers.Dense(10)(X)  # The 10 here is the number of output classes

    X = keras.layers.Dropout(0.3)(X)

    

    # Use softmax to get the output to represent a probability distribution

    X = keras.layers.Activation('softmax')(X)

    

    model = keras.models.Model(inputs=X_input, outputs=X, name='MNIST_model')

    model.compile(

        optimizer=keras.optimizers.Adam(),

        loss=keras.losses.categorical_crossentropy,

        metrics=[keras.metrics.categorical_accuracy],

    )

    return model



model = get_model()

print(model)
model.fit(

    x=data, 

    y=labels, 

    batch_size=32, 

    epochs=10, 

    callbacks=[keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=5)]

)
predictions = model.predict(x=test_data)
categorical_predictions = predictions.argmax(axis=-1)

with open('predictions.csv', 'w') as outfile:

    outfile.write('ImageId,Label\n')

    for i, number in enumerate(categorical_predictions):

        outfile.write(f'{i+1},{number}\n')
with open('predictions.csv') as infile:

    for line in infile:

        print(line, end='')
from keras import Input

from keras.datasets import mnist

from keras.utils import to_categorical

from keras.models import Model, Sequential, load_model, save_model

from keras.layers import Conv2D, Flatten, Dense



from numpy import ndarray

from os.path import isfile



def get_functional_model() -> Model:

    if not force and isfile(functional_filename):

        print('Loading functional model')

        model = load_model(functional_filename)

    else:

        print('Creating functional model')



        inputLayer = Input(shape=(image_size, image_size, 1))



        convolutionalFirstLayer = Conv2D(64, kernel_size=3, activation='relu')(inputLayer)

        convolutionalSecondLayer = Conv2D(32, kernel_size=3, activation='relu')(convolutionalFirstLayer)

        flattenLayer = Flatten()(convolutionalSecondLayer)

        outputLayer = Dense(10, activation='softmax')(flattenLayer)



        model = Model(inputs=[inputLayer], outputs=[outputLayer])



        print('Compiling fucntional model now')



        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



        print('Fitting fucntional model now.')



        model.fit(x_train, y_train, validation_split=0, epochs=1)



    return model



def get_sequential_model()-> Model:

    if not force and isfile(functional_filename):

        print('Loading sequential model')

        model = load_model(sequential_filename)

    else:

        print('Creating sequential model')



        model = Sequential()



        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(image_size, image_size, 1)))

        model.add(Conv2D(32, kernel_size=3, activation='relu'))

        model.add(Flatten())

        model.add(Dense(10, activation='softmax'))



        print('Compiling sequential model now')



        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



        print('Fitting sequential model now.')



        model.fit(x_train, y_train, validation_split=0, epochs=1)



    return model



image_size = 28



force = False

sequential_filename = 'sequential.h5'

functional_filename = 'functional.h5'



(original_x_train, y_train), (original_x_test, y_test) = mnist.load_data()



x_train: ndarray = original_x_train.reshape(60000, image_size, image_size, 1)

x_test: ndarray = original_x_test.reshape(10000, image_size, image_size, 1)



#one-hot encode target column

y_train = to_categorical(y_train)

y_test = to_categorical(y_test)



functional_model = get_functional_model()

sequential_model = get_sequential_model()



if force or not isfile(functional_filename):

    print('Saving functional model')

    save_model(functional_model, functional_filename)



if force or not isfile(sequential_filename):

    print('Saving sequential model')

    save_model(sequential_model, sequential_filename)



print('Predicting on batch now.')



functional_results = {}

sequential_results = {}



for i, result in enumerate(functional_model.predict_on_batch(x_test)):

    target = y_test[i]



    predicted = None

    predictedIndex = None



    for j, value in enumerate(result):

        if predicted is None or value > predicted:

            predicted = value

            predictedIndex = j



    actual = None

    actualIndex = None

    for j, value in enumerate(target):

        if actual is None or value > actual:

            actual = value

            actualIndex = j



    if actualIndex not in functional_results:

        functional_results[actualIndex] = {

            'correct': 0,

            'incorrect': 0

        }



    if predictedIndex == actualIndex:

        functional_results[actualIndex]['correct'] += 1

    else:

        functional_results[actualIndex]['incorrect'] += 1



for i, result in enumerate(sequential_model.predict_on_batch(x_test)):

    target = y_test[i]



    predicted = None

    predictedIndex = None



    for j, value in enumerate(result):

        if predicted is None or value > predicted:

            predicted = value

            predictedIndex = j



    actual = None

    actualIndex = None

    for j, value in enumerate(target):

        if actual is None or value > actual:

            actual = value

            actualIndex = j



    if actualIndex not in sequential_results:

        sequential_results[actualIndex] = {

            'correct': 0,

            'incorrect': 0

        }



    if predictedIndex == actualIndex:

        sequential_results[actualIndex]['correct'] += 1

    else:

        sequential_results[actualIndex]['incorrect'] += 1



for index in range(len(functional_results)):

    functionalData = functional_results[index]

    sequentialData = sequential_results[index]



    functionalProportionCorrect = (functionalData['correct'] / (functionalData['correct'] + functionalData['incorrect']))

    sequentialProportionCorrect = (sequentialData['correct'] / (sequentialData['correct'] + sequentialData['incorrect']))



    print(f'''Index {index}

Sequential {sequentialProportionCorrect * 100}%

Functional {functionalProportionCorrect * 100}%\n''')
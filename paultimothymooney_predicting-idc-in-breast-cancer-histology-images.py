import numpy as np

import matplotlib.pylab as plt

from scipy.misc import imresize, imread

import itertools

import sklearn

from sklearn import model_selection

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV

from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import keras

from keras import backend as K

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

from keras.models import Sequential, model_from_json

from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D

%matplotlib inline
X = np.load('../input/X.npy') # images

Y = np.load('../input/Y.npy') # labels associated to images (0 = no IDC, 1 = IDC)
def describeData(a,b):

    print('Total number of images: {}'.format(len(a)))

    print('Number of IDC(-) Images: {}'.format(np.sum(b==0)))

    print('Number of IDC(+) Images: {}'.format(np.sum(b==1)))

    print('Percentage of positive images: {:.2f}%'.format(100*np.mean(b)))

    print('Image shape (Width, Height, Channels): {}'.format(a[0].shape))

describeData(X,Y)
imgs0 = X[Y==0] # (0 = no IDC, 1 = IDC)

imgs1 = X[Y==1] 



def plotOne(a,b):

    """

    Plot one numpy array

    """

    plt.subplot(1,2,1)

    plt.title('IDC (-)')

    plt.imshow(a[100])

    plt.subplot(1,2,2)

    plt.title('IDC (+)')

    plt.imshow(b[100])

plotOne(imgs0, imgs1) 

def plotTwo(a,b): 

    """

    Plot a bunch of numpy arrays sorted by label

    """

    for row in range(3):

        plt.figure(figsize=(20, 10))

        for col in range(3):

            plt.subplot(1,8,col+1)

            plt.title('IDC (-)')

            plt.imshow(a[row+col])

            plt.axis('off')       

            plt.subplot(1,8,col+4)

            plt.title('IDC (+)')

            plt.imshow(b[row+col])

            plt.axis('off')

plotTwo(imgs0, imgs1) 

def plotHistogram(a):

    """

    Plot histogram of RGB Pixel Intensities

    """

    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)

    plt.imshow(a)

    plt.axis('off')

    plt.title('IDC(+)' if Y[1] else 'IDC(-)')

    histo = plt.subplot(1,2,2)

    histo.set_ylabel('Count')

    histo.set_xlabel('Pixel Intensity')

    n_bins = 30

    plt.hist(a[:,:,0].flatten(), bins= n_bins, lw = 0, color='r', alpha=0.5);

    plt.hist(a[:,:,1].flatten(), bins= n_bins, lw = 0, color='g', alpha=0.5);

    plt.hist(a[:,:,2].flatten(), bins= n_bins, lw = 0, color='b', alpha=0.5);

plotHistogram(X[100])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



# Reduce Sample Size for DeBugging

X_train = X_train[0:30000] 

Y_train = Y_train[0:30000]

X_test = X_test[0:30000] 

Y_test = Y_test[0:30000]



# Normalize the data

X_train = X_train / 256.0

X_test = X_test / 256.0



print("Training Data Shape:", X_train.shape, X_train.shape)

print("Testing Data Shape:", X_test.shape, X_test.shape)

plotHistogram(X_train[100])
# Make Data 1D for compatability with standard classifiers



X_trainShape = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]

X_testShape = X_test.shape[1]*X_test.shape[2]*X_test.shape[3]



X_trainFlat = X_train.reshape(X_train.shape[0], X_trainShape)

X_testFlat = X_test.reshape(X_test.shape[0], X_testShape)
#runLogisticRegression

def runLogisticRegression(a,b,c,d):

    """Run LogisticRegression w/ Kfold CV"""

    model = LogisticRegression()

    model.fit(a,b)

    kfold = model_selection.KFold(n_splits=10)

    accuracy = model_selection.cross_val_score(model, c,d, cv=kfold, scoring='accuracy')

    mean = accuracy.mean() 

    stdev = accuracy.std()

    print('LogisticRegression - Training set accuracy: %s (%s)' % (mean, stdev))

    print('')

runLogisticRegression(X_trainFlat, Y_train, X_testFlat, Y_test)
# Compare Performance of Classification Algorithms

def compareABunchOfDifferentModelsAccuracy(a,b,c,d):

    """

    compare performance of classifiers on X_train, X_test, Y_train, Y_test

    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

    http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score

    """    

    print('')

    print('Compare Multiple Classifiers:')

    print('')

    print('K-Fold Cross-Validation Accuracy:')

    print('')

    models = []

    models.append(('LR', LogisticRegression()))

    models.append(('RF', RandomForestClassifier()))

    models.append(('KNN', KNeighborsClassifier()))

    models.append(('SVM', SVC()))

    models.append(('LSVM', LinearSVC()))

    models.append(('GNB', GaussianNB()))

    models.append(('DTC', DecisionTreeClassifier()))

    #models.append(('GBC', GradientBoostingClassifier()))

    #models.append(('LDA', LinearDiscriminantAnalysis()))       

    resultsAccuracy = []

    names = []

    for name, model in models:

        model.fit(a, b)

        kfold = model_selection.KFold(n_splits=10)

        accuracy_results = model_selection.cross_val_score(model, c, d, cv=kfold, scoring='accuracy')

        resultsAccuracy.append(accuracy_results)

        names.append(name)

        accuracyMessage = "%s: %f (%f)" % (name, accuracy_results.mean(), accuracy_results.std())

        print(accuracyMessage)  

    # boxplot algorithm comparison

    fig = plt.figure()

    fig.suptitle('Algorithm Comparison: Accuracy')

    ax = fig.add_subplot(111)

    plt.boxplot(resultsAccuracy)

    ax.set_xticklabels(names)

    ax.set_ylabel('Cross-Validation: Accuracy Score')

    plt.show()

    return

compareABunchOfDifferentModelsAccuracy(X_trainFlat, Y_train, X_testFlat, Y_test)





def defineModels():

    """

    This function just defines each abbreviation used in the previous function (e.g. LR = Logistic Regression)

    """

    print('')

    print('LR = LogisticRegression')

    print('RF = RandomForestClassifier')

    print('KNN = KNeighborsClassifier')

    print('SVM = Support Vector Machine SVC')

    print('LSVM = LinearSVC')

    print('GNB = GaussianNB')

    print('DTC = DecisionTreeClassifier')

    #print('GBC = GradientBoostingClassifier')

    #print('LDA = LinearDiscriminantAnalysis')

    print('')

    return

defineModels()
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Plots a learning curve. http://scikit-learn.org/stable/modules/learning_curve.html

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

    plt.legend(loc="best")

    return plt



def plotLotsOfLearningCurves(a,b):

    """Plot a bunch of learning curves http://scikit-learn.org/stable/modules/learning_curve.html"""

    models = []

    models.append(('Support Vector Machine', SVC()))

    for name, model in models:

        plot_learning_curve(model, 'Learning Curve For %s Classifier'% (name), a,b, (0.5,1), 10)

plotLotsOfLearningCurves(X_trainFlat, Y_train)
# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.figure(figsize = (5,5))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=90)

    plt.yticks(tick_marks, classes)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



#Run SVC w/ Confusion Matrix

def runSVCconfusion(a,b,c,d):

    """Run SVC w/ Kfold CV + Confusion Matrix"""

    model = SVC()

    model.fit(a, b)

    prediction = model.predict(c)

    kfold = model_selection.KFold(n_splits=10)

    accuracy = model_selection.cross_val_score(model, c,d, cv=kfold, scoring='accuracy')

    mean = accuracy.mean() 

    stdev = accuracy.std()

    print('\nSupport Vector Machine - Training set accuracy: %s (%s)' % (mean, stdev),"\n")

    cnf_matrix = confusion_matrix(d, prediction)

    np.set_printoptions(precision=2)

    class_names = ["Diagnosis" "IDC(-)", "Diagnosis" "IDC(+)"]

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=class_names,

                          title='Confusion matrix, without normalization')

    plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,

                          title='Normalized confusion matrix')

    plt.show()

runSVCconfusion(X_trainFlat, Y_train, X_testFlat, Y_test)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])

Y_train = to_categorical(Y_train, num_classes = 2)

Y_test = to_categorical(Y_test, num_classes = 2)
# Special callback to see learning curves

class MetricsCheckpoint(Callback):

    """Callback that saves metrics after each epoch"""

    def __init__(self, savepath):

        super(MetricsCheckpoint, self).__init__()

        self.savepath = savepath

        self.history = {}

    def on_epoch_end(self, epoch, logs=None):

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        np.save(self.savepath, self.history)



def plotKerasLearningCurve():

    plt.figure(figsize=(10,5))

    metrics = np.load('logs.npy')[()]

    filt = ['acc'] # try to add 'loss' to see the loss learning curve

    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):

        l = np.array(metrics[k])

        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')

        x = np.argmin(l) if 'loss' in k else np.argmax(l)

        y = l[x]

        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')

        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')   

    plt.legend(loc=4)

    plt.axis([0, None, None, None]);

    plt.grid()

    plt.xlabel('Number of epochs')

def runKerasCNN(a,b,c,d):

    """

    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

    """

    batch_size = 128

    num_classes = 2

    epochs = 12  

    img_rows, img_cols = X_train.shape[1],X_train.shape[2]

    input_shape = (img_rows, img_cols, 3)

    x_train = a

    y_train = b

    x_test = c

    y_test = d   

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer=keras.optimizers.Adadelta(),

                  metrics=['accuracy'])

    model.fit(x_train, y_train,

              batch_size=batch_size,

              verbose=1,

              epochs=epochs,

              validation_data=(x_test, y_test),callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(x_test, y_test, verbose=0)

    print('\nKeras CNN #1A - accuracy:', score[1],'\n')

    y_pred = model.predict(c) 

    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(Y_test,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 

runKerasCNN(X_train, Y_train,  X_test, Y_test)

plotKerasLearningCurve()
def runAnotherKeras(a, b,c,d):

    # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out   

    batch_size = 128

    num_classes = 2

    epochs = 12

    # input image dimensions

    img_rows, img_cols = X_train.shape[1],X_train.shape[2]

    input_shape = (img_rows, img_cols, 3)

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu', input_shape = input_shape))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Flatten())

    #model.add(Dense(1024, activation = "relu"))

    #model.add(Dropout(0.5))

    model.add(Dense(512, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation = "softmax"))

    optimizer = RMSprop(lr=0.001, decay=1e-6)

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    model.fit(a,b,

                  batch_size=batch_size,

                  verbose=1,

                  epochs=epochs,

                  validation_data=(c,d),callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(c,d, verbose=0)

    print('\nKeras CNN #2 - accuracy:', score[1], '\n')

    y_pred = model.predict(c)

    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='') 

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(Y_test,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 

runAnotherKeras(X_train, Y_train, X_test, Y_test)

plotKerasLearningCurve()
def kerasAugmentation(a,b,c,d):

    img_rows, img_cols = 50,50

    input_shape = (img_rows, img_cols, 3)

    batch_size = 128

    num_classes = 2

    epochs = 12

    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape))

    model.add(Activation('relu'))

    model.add(Conv2D(32, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same'))

    model.add(Activation('relu'))

    model.add(Conv2D(64, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same')) 

    model.add(Activation('relu'))

    model.add(Conv2D(256, (3, 3)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1024))

    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

    datagen.fit(a)

    model.fit_generator(datagen.flow(a,b, batch_size=32),

                        steps_per_epoch=len(a) / 32, epochs=epochs, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(c,d, verbose=0)

    print('\nKeras CNN #3B - accuracy:', score[1],'\n')

    y_pred = model.predict(c)

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(Y_test,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 

kerasAugmentation(X_train, Y_train, X_test, Y_test)

plotKerasLearningCurve()
def runAnotherKerasAugmentedConfusion(a,b,c,d):

    # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out 

    batch_size = 128

    num_classes = 2

    epochs = 16

    img_rows, img_cols = X_train.shape[1],X_train.shape[2]

    input_shape = (img_rows, img_cols, 3)

    model = Sequential() 

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu', input_shape = input_shape))

    model.add(Conv2D(filters = 32, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(Conv2D(filters = 86, kernel_size = (3,3),padding = 'Same', 

                     activation ='relu'))

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(BatchNormalization())

    model.add(Dropout(0.25))

    model.add(Flatten())

    #model.add(Dense(1024, activation = "relu"))

    #model.add(Dropout(0.5))

    model.add(Dense(512, activation = "relu"))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation = "softmax"))

    optimizer = RMSprop(lr=0.001, decay=1e-6)

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

    datagen.fit(a)

    model.fit_generator(datagen.flow(a,b, batch_size=32),steps_per_epoch=len(a) / 32, epochs=epochs, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(c,d, verbose=0)

    print('\nKeras CNN #2B - accuracy:', score[1],'\n')

    y_pred = model.predict(c)

    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(Y_test,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 

runAnotherKerasAugmentedConfusion(X_train, Y_train, X_test, Y_test)   

plotKerasLearningCurve()
# Create the model

def yetAnotherKeras(a,b,c,d):

    model = Sequential()

    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(50, 50, 3))) # first layer : convolution

    model.add(MaxPooling2D(pool_size=(3, 3))) # second layer : pooling (reduce the size of the image per 3) 

    model.add(Conv2D(32, (5, 5), activation='relu')) 

    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid')) # output 1 value between 0 and 1 : probability to have cancer

    model.summary()

    model.compile(loss=keras.losses.binary_crossentropy, # Use binary crossentropy as a loss function  

                  optimizer=keras.optimizers.Adam(),

                  metrics=['accuracy'])

    model.fit(a,b,

              batch_size=128,

              epochs=12,

              verbose=1,

              validation_data = [c,d],

            callbacks = [MetricsCheckpoint('logs')])

    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

    y_pred = model.predict(c)

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(Y_test,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 

yetAnotherKeras(X_train,Y_train,X_test,Y_test)

plotKerasLearningCurve()
def runKerasCNNAugment(a,b,c,d):

    """

    Run Keras CNN: https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

    """

    batch_size = 128

    num_classes = 2

    epochs = 12

    img_rows, img_cols = X_train.shape[1],X_train.shape[2]

    input_shape = (img_rows, img_cols, 3)

    x_train = a

    y_train = b

    x_test = c

    y_test = d

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer=keras.optimizers.Adadelta(),

                  metrics=['accuracy'])

    datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=True,  # randomly flip images

        vertical_flip=False)  # randomly flip images

    model.fit_generator(datagen.flow(a,b, batch_size=32),

                        steps_per_epoch=len(a) / 32, epochs=epochs, validation_data = [c, d],callbacks = [MetricsCheckpoint('logs')])

    score = model.evaluate(c,d, verbose=0)

    print('\nKeras CNN #1C - accuracy:', score[1],'\n')

    y_pred = model.predict(c)

    map_characters = {0: 'IDC(-)', 1: 'IDC(+)'}

    print('\n', sklearn.metrics.classification_report(np.where(d > 0)[1], np.argmax(y_pred, axis=1), target_names=list(map_characters.values())), sep='')    

    score = model.evaluate(x_test, y_test, verbose=0)

    Y_pred_classes = np.argmax(y_pred,axis = 1) 

    Y_true = np.argmax(Y_test,axis = 1) 

    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

    plot_confusion_matrix(confusion_mtx, classes = list(map_characters.values())) 

runKerasCNNAugment(X_train, Y_train,  X_test, Y_test)

plotKerasLearningCurve()
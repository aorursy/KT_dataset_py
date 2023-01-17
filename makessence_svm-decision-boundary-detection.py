import numpy as np

import matplotlib.pyplot as plt

import h5py

from PIL import Image

from scipy import ndimage

from sklearn import preprocessing, svm

from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier

np.random.seed(1)
def load_dataset():

    train_dataset = h5py.File('../input/cat-images-dataset/train_catvnoncat.h5', "r")

    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features

    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels



    test_dataset = h5py.File('../input/cat-images-dataset/test_catvnoncat.h5', "r")

    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features

    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels



    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

x_train, y_train, x_test, y_test, classes = load_dataset()

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

x_train=x_train/255

x_test=x_test/255
#Some functions to show images

def show_img(x, plotter=plt):

    plotter.imshow(x.reshape(64, 64, 3))

    

def plot_grid(arr, grid_size):

    cols, rows = grid_size

    f, plots = plt.subplots(rows, cols, figsize = (cols * 4, rows * 4))

    plots = plots.reshape(rows * cols)

    for row, plot in zip(arr, plots):

        show_img(row, plot)

        plot.set_axis_off()

    

plot_grid(x_train, (5, 5))
#reshaping arrays

x_train=x_train.reshape((209,64*64*3))

y_train = y_train.reshape((209))



x_test=x_test.reshape((50,64*64*3))

y_test = y_test.reshape((50))
params = {'kernel':['linear'], 'C':[0.01, 0.1, 1, 10, 100, 1000]}

classifier = GridSearchCV(svm.SVC(kernel='linear'), params, scoring='f1', cv=5) # Using 5-fold cross validation to find best value of C
classifier.fit(x_train, y_train)
svm_classifier = classifier.best_estimator_

print('Training set accuracy', svm_classifier.score(x_train, y_train))

print('Test set accuracy', svm_classifier.score(x_test, y_test))
# We know shortest direction from any point to the plane (normal vector w)

# And we can calculate distance to the plane using formula d = w*x + b

# So we can move a point twice the distance in order to reflect it

def reflect_across_plane(x, w, b, multiplier=2):

    normal_len = w @ w.T

    dist_to_plane = x @ w.T + b

    reflected_x = x - w * (multiplier * dist_to_plane / normal_len)

    return np.clip(reflected_x, 0, 1) # To make sure that values are still in range
# Extracting hyperplane parameters from model

w = svm_classifier.coef_

b = svm_classifier.intercept_
# Sanity check

# Should be equal to (1 - Test set accuracy)

print('Reflected accuracy', svm_classifier.score(reflect_across_plane(x_test, w, b), y_test))
# NN with two inner layers of 1000 neurons each

nn_classifier = MLPClassifier((1000, 1000))
nn_classifier.fit(x_train, y_train)
print('Training set accuracy', nn_classifier.score(x_train, y_train))

print('Test set accuracy', nn_classifier.score(x_test, y_test))

print('Reflected accuracy', nn_classifier.score(reflect_across_plane(x_test, w, b), y_test))
accuaracies_reflected = [nn_classifier.score(reflect_across_plane(x_test, w, b, i), y_test) for i in range(10)]



# Reflected dataset accuracy was somehow even better than original

# So let's only reflect points that are classified correctly by our svm

# And just move others in the opposite direction

multipliers = ((y_test == svm_classifier.predict(x_test)) * 2 - 1).reshape(50, 1) # 1 if point on the right side of SVMs hyperplane, -1 otherwise



accuaracies_reflected_conditionally = [nn_classifier.score(reflect_across_plane(x_test, w, b, (multipliers * i)), y_test) for i in range(10)]
plt.plot(range(10), accuaracies_reflected, 'r', range(10), accuaracies_reflected_conditionally, 'b')

plt.xlabel('distance moved')

plt.ylabel('accuracy')
# Increasing ditance from the hyperplane decreases classification accuracy but also makes images more noisy



rows = (0, 1, 5, 16)

vectors = []

for i in range(10):

    for row in rows:

        vectors.append(reflect_across_plane(x_test[row], w, b, multipliers[row] * i))



    

plot_grid(vectors, (len(rows), 10))
specialized_classifier = GridSearchCV(svm.SVC(kernel='linear'), params, scoring='f1', cv=5)

specialized_classifier.fit(x_train, nn_classifier.predict(x_train))
# Parameters from new model

specialized_svm_classifier = specialized_classifier.best_estimator_



w2 = specialized_svm_classifier.coef_

b2 = specialized_svm_classifier.intercept_
# Using conditional approach from the start

multipliers = ((y_test == specialized_svm_classifier.predict(x_test)) * 2 - 1).reshape(50, 1)

accuaracies_reflected_specialized = [nn_classifier.score(reflect_across_plane(x_test, w2, b2, (multipliers * i)), y_test) for i in range(10)]



plt.plot(range(10), accuaracies_reflected, 'b', range(10), accuaracies_reflected_specialized, 'y')
multipliers = ((y_train == svm_classifier.predict(x_train)) * 2 - 1).reshape(209, 1)



accuracies_poisoned = []



for sample_size in np.arange(0, 1, 0.1):

    accuracy_accumulator = 0

    for i in range(10):

        mask = np.zeros((209, 1))

        mask[0: int(sample_size * 209)] = 2

        np.random.shuffle(mask)

        poisoned_x_train = reflect_across_plane(x_train, w, b, (multipliers * mask))

    

        poisoned_nn_classifier = MLPClassifier((100, 100))

        poisoned_nn_classifier.fit(poisoned_x_train, y_train)

        

        accuracy_accumulator += poisoned_nn_classifier.score(x_test, y_test)

    accuracies_poisoned.append(accuracy_accumulator / 10)
plt.plot(np.arange(0, 1, 0.1), accuracies_poisoned)
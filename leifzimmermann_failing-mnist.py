import numpy as np

import pandas as pd

import os

import sys

import time

import random

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt



# import sklearn functions

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, ParameterGrid

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.utils import shuffle

from sklearn.metrics import precision_score, recall_score, accuracy_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder,LabelEncoder, LabelBinarizer

from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.kernel_approximation import Nystroem

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier

from sklearn.svm import SVC, LinearSVC



# other stuff needed

from skimage.transform import resize

from scipy.stats import uniform, expon, randint

from scipy.ndimage.interpolation import shift, rotate
# the next two functions are taken from Aurelien Gerons book "Hands-On Machine Learning 

# with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques for Building Intelligent Systems",

# from the notebook https://github.com/ageron/handson-ml2/blob/master/03_classification.ipynb

def plot_digit(data, cmap='gray', size=28):

    image = data.reshape(size, size)

    plt.imshow(image, cmap = cmap,interpolation="nearest")

    #plt.axis("off")



def plot_digits(instances, images_per_row=10, size=28,**options):

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = 'gray', **options) # cmap = mpl.cm.binary

    plt.axis("off")



def fit_time_est(estimator, X_train, y_train, prnt=True):

    start = time.time()

    estimator.fit(X_train, y_train)

    end = time.time()

    if prnt:

        print("Time needed to fit the estimator:", end - start)

    return end - start



def fit_score_time(estimator, X_train, y_train, X_val, y_val, prnt=True, oob=False):

    time_fit = fit_time_est(estimator, X_train, y_train, prnt=False)

    if oob:

        if prnt:

            print("Time needed for fitting:", time_fit)

            print("Estimate of generalization error from oob (out of bag samples):", estimator.oob_score_)

        #return est, time_fit, est.oob_score_

    else:    

        start = time.time()

        y_pred = estimator.predict(X_val)

        acc_score = accuracy_score(y_pred,y_val)

        end = time.time()

        if prnt:

            print("Time needed for fitting:", time_fit)

            print("Valuation score:", acc_score, " with", end - start, "seconds needed to evaluate.")

    

def crop_inst(X_inst, tr_im_size=[28,28], crop_left=3, crop_right=3, crop_top=3, crop_bottom=3):

    """crop_data a data instance into its image_size, crops the given pixels from the border

    and transforms the new image back into a row array"""

    X_im = X_inst.reshape(tr_im_size)

    X_im = X_im[crop_top:-crop_bottom, crop_left:-crop_right] # do the cropping

    return X_im.reshape((tr_im_size[0]-crop_top-crop_bottom)*(tr_im_size[1]-crop_left-crop_right))

    



def crop_data(X_train, tr_im_size=[28,28], crop_left=3, crop_right=3, crop_top=3, crop_bottom=3):

    """crop_data turns every data instance into its image_size, crops the given pixels from the border

    and transforms the new image back into a row array, returning thus a new data set"""

    return np.apply_along_axis(crop_inst,1,X_train,tr_im_size=tr_im_size,crop_left=crop_left,

                               crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom)



def instance_resizer(instance, orig_im_size=(28,28), new_im_size=(14,14), mode='reflect', anti_aliasing=True):

    """takes a vector, interprets it as a picture, resizes it and returns the new image as a vector"""

    image_orig = instance.reshape(orig_im_size)

    image_new = resize(image_orig, new_im_size, mode=mode)

    inst_transformed = image_new.reshape(new_im_size[0]*new_im_size[1])

    return inst_transformed



def resize_train(X_train, orig_im_size=(28,28), new_im_size=(14,14), mode='reflect',anti_aliasing=True):

    """applies instance_resizer to every instance"""

    return np.apply_along_axis(instance_resizer,1,X_train,orig_im_size=orig_im_size,new_im_size=new_im_size,

                               mode=mode, anti_aliasing=anti_aliasing)



def stratify_data(X_train, threshold=230):

    return np.where(X_train>threshold/255, 1, 0).astype(np.bool) # the /255 comes from a former version where the pixels where not normalized



def pollute_instance(instance, im_size=(28,28), n_poll_pic=1, random_state = None):

    if not random_state is None:

        np.random.seed(random_state)

    inst = instance.copy() # otherwise, we change the original data

    image = inst.reshape(im_size)

    # set rectangle with pollution

    for _ in range(n_poll_pic):

        a, b = np.random.randint(0,im_size[0]), 1+int(4*np.random.uniform())

        c, d = np.random.randint(0,im_size[1]), 1+int(4*np.random.uniform())

        # start polluting

        for i in range(a,a+b):

            for j in range(c,c+d):

                try:

                    image[i,j] = np.random.randint(50,200)/255 # /255 is added because pixels are assumed to be in [0,1]

                except IndexError:

                    pass

    inst_transformed = image.reshape(im_size[0]*im_size[1])



    return inst_transformed

    



def pollute_training(X_train, im_size=(28,28), n_poll_pic=1, random_state = None):

    # usually, we would have used the following code, but it does not work with random_state set because in each

    # instance the same pixels are polluted

    if random_state is None:

        return np.apply_along_axis(pollute_instance,1,X_train,im_size=im_size,

                                   n_poll_pic=n_poll_pic, random_state=random_state)

    else:

        X_pol = X_train.copy()

        for i in range(X_train.shape[0]):

            X_pol[i] = pollute_instance(X_train[i],im_size=im_size, n_poll_pic=n_poll_pic,

                                        random_state=random_state+i)

        return X_pol



# shift_instance and rand_shift_training are adapted versions from Aurelien Gerons book chapter 3 additional material

# see his notebooks

# to avoid combinatorial explosion, we use random shifting and rotating



def shift_instance(instance, dx, dy, new=0, im_size=(28,28)):

    return shift(instance.reshape(im_size), [dy, dx], cval=new).reshape(im_size[0]*im_size[1])



def rand_shift_training(X_train, y_train, dx_max=2, dy_max=2, n_shifts = 4, new=0,

                        im_size=(28,28), random_state=None, only_axes=True, do_shuffle=True):

    

    if not random_state is None:

        np.random.seed(random_state)

    

    # collect all possibilities to move an image with given dx_max, dy_max

    moving_vectors = []

    if only_axes:

        for i in range(1,dx_max+1): # if i=0 or j=0, we will get double entries. we will remove them below

            moving_vectors.append((i,0))

            moving_vectors.append((-i,0))

        for i in range(1,dy_max+1):

            moving_vectors.append((0,i))

            moving_vectors.append((0,-i))

    else:

        for i in range(0,dx_max+1):

            for j in range(0,dy_max+1): # if i=0 or j=0, we will get double entries. we will remove them below

                moving_vectors.append((i,j))

                moving_vectors.append((-i,j))

                moving_vectors.append((i,-j))

                moving_vectors.append((-i,-j))

    

    moving_vectors = list(set(moving_vectors)) # removes double entries from moving_vectors



    # we do not want to double images, so remove (0,0) from moving_vectors

    try:

        moving_vectors.remove((0,0))

    except ValueError:

        if not only_axes: # because there should be no (0,0) inside if only_axes is True

            print("No (0,0) could be removed from moving_vectors. Please check the code.")

    

    

    X_train_expanded = []

    y_train_expanded = []

    

    # move image by moving_vectors and add to X_train_expanded

    for i in range(X_train.shape[0]):

        try:

            for dx, dy in random.sample(moving_vectors, n_shifts):

                shifted_image = shift_instance(X_train[i], dx=dx, dy=dy, new=0, im_size=im_size)

                X_train_expanded.append(shifted_image)

                y_train_expanded.append(y_train[i])

        except ValueError:

            print("Using maximal possible shifts. This might be smaller thatn n_shifts.")

            for dx, dy in moving_vectors:

                shifted_image = shift_instance(X_train[i], dx=dx, dy=dy, new=0, im_size=im_size)

                X_train_expanded.append(shifted_image)

                y_train_expanded.append(y_train[i])



    X_train_expanded = np.array(X_train_expanded, dtype=np.uint8)

    y_train_expanded = np.array(y_train_expanded, dtype=np.uint8)

    

    # shuffle the new shifted images to avoid classifiers see the same number for too many times

    if do_shuffle:

        X_train_expanded, y_train_expanded = shuffle(X_train_expanded, y_train_expanded, random_state=random_state)

    return X_train_expanded, y_train_expanded



def rand_rotate_instance(instance, angle=10, new=0, im_size=(28,28)):

    return rotate(instance.reshape(im_size), angle=angle, cval=new, reshape=False).reshape(im_size[0]*im_size[1])



def rand_rotate_training(X_train, y_train, angle_range=(5,10), im_size=(28,28),

                        sign=True, do_shuffle=True, random_state=None):

    

    if not random_state is None:

        np.random.seed(random_state)

    

    X_rot = X_train.copy()

    for i in range(X_train.shape[0]): # we need this loop to not always get the same rotation

        angle = np.random.randint(angle_range[0],angle_range[1])

        X_rot[i] = rand_rotate_instance(X_train[i],angle=angle,im_size=im_size)

    

    if sign:

        X_rot_sign = X_train.copy()

        for i in range(X_train.shape[0]): # we need this loop to not always get the same rotation

            angle = np.random.randint(angle_range[0],angle_range[1])

            X_rot_sign[i] = rand_rotate_instance(X_train[i],angle=-angle,im_size=im_size)

        

        # shuffle the new shifted images to avoid classifiers see the same number for too many times

        if do_shuffle:

            X_train_expanded, y_train_expanded = shuffle(np.concatenate((X_rot, X_rot_sign)),

                                                         np.concatenate((y_train, y_train)),

                                                         random_state=random_state)

        return X_train_expanded, y_train_expanded    

    else:

        return X_rot, y_train

    

def augment_training(X_train, y_train,dx_max=2,dy_max=2,n_shifts = 4,angle_range=(5,10),im_size=(28,28),

                     only_axes=False, do_shuffle=True, sign=True, random_state=None):

    X_shift, y_shift = rand_shift_training(X_train, y_train, dx_max=dx_max, dy_max=dy_max, n_shifts = n_shifts, 

                        im_size=im_size, random_state=random_state, only_axes=only_axes, do_shuffle=do_shuffle)

    X_rot, y_rot = rand_rotate_training(X_train, y_train, angle_range=angle_range, im_size=im_size,

                        sign=sign, do_shuffle=do_shuffle, random_state=random_state)

    X_train = np.concatenate((X_train,X_shift, X_rot))

    y_train = np.concatenate((y_train, y_shift, y_rot))

    return X_train, y_train
class ImageTransformer(BaseEstimator, TransformerMixin): # the arguments say that the new class inherits from the others, namely get/setparams_and fit_transform

    def __init__(self, resize=True, pollute=True, crop=True, stratify=True, n_poll_pic=1, random_state=None): # pass only whether to apply some transformations and random state

        self.resize = resize

        self.crop = crop

        self.stratify = stratify

        self.pollute = pollute

        self.n_poll_pic = n_poll_pic

        self.random_state = random_state

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X): # the X should be an array accessible by dataframe.values. Important as pipeline only handles these guys

        if self.pollute:

            X = pollute_training(X,n_poll_pic = self.n_poll_pic, random_state=self.random_state)

        if self.resize:

            if X.shape[1] != 14*14: # only resize if pictures are not already resized to (14,14)

                X = resize_train(X)

        if self.crop:

            X = crop_data(X, tr_im_size=[14,14], crop_left=1, crop_right=1, crop_top=1, crop_bottom=1)

        if self.stratify:

            stratify_data(X, threshold=0)

        return X
# load the data

df_train = pd.read_csv("../input/digit-recognizer/train.csv", engine='c')

df_test = pd.read_csv("../input/digit-recognizer/test.csv", engine='c')
# inspect the data

df_train
# check for completeness

# only labels in the range of 0..9 ???

df_train['label'].value_counts()
# so the labels appear more or less in the same frequence. in particular, accuracy score is fine

df_train['label'].hist()
# check for null values

df_train.isnull().any().describe()
# no null value. great. lets start by taking the numpy arrays and splitting the label from the instances

y = df_train['label'].to_numpy(dtype = np.uint8)

X = df_train.drop(labels = ["label"],axis = 1).to_numpy()/255
# lets check sizes

print(y.shape)

print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=5000, random_state=43, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size=5000, random_state=43, stratify=y_train)
lab_bin = LabelBinarizer()

y_train_label = lab_bin.fit_transform(y_train)

y_val_label = lab_bin.transform(y_val) # dont fit again, so that the correspondance of digit and list keeps the same
# lets have a look how it works:

print(y_train[:10])

y_train_label[:10]
# lets see pictures

plt.figure(figsize=(9,9))

example_images = X_train[:100]

plot_digits(example_images, images_per_row=10)

#save_fig("more_digits_plot")

plt.show()
plot_digit(X_train[0])
# lets check the performance of a random forest as a bar

# they are rather fast, so we can apply the whole training set

rdf_clf = RandomForestClassifier(random_state=43, oob_score=True, n_jobs=-1) # n_jobs=-1 takes all processors



fit_score_time(rdf_clf, X_train, y_train, X_val, y_val, prnt=True, oob=True) # took 23 seconds, score 0.958
fit_score_time(rdf_clf, X_train, y_train_label, X_val, y_val, prnt=True, oob=True) # took 45 seconds,score 0.986
# as we can see, a great part of the picture is unimportant as we already saw

plot_digit(rdf_clf.feature_importances_, cmap=mpl.cm.hot)
# WARNING: this code took about 150 seconds on my machine

knn = KNeighborsClassifier()

sgd = make_pipeline(StandardScaler(),SGDClassifier(random_state=43))

x_trees = ExtraTreesClassifier(random_state=43)

svc_rbf = make_pipeline(StandardScaler(),SVC(kernel='rbf', random_state=43)) #  kernel='rbf' is the default in sklearn

lin_svc = make_pipeline(StandardScaler(),LinearSVC(random_state=43))

grad_boost = GradientBoostingClassifier(random_state=43)



classifiers = [knn, sgd,x_trees,svc_rbf,lin_svc, grad_boost]



start = time.time()

for clf in classifiers:

    print(clf.__class__.__name__)

    for i in [100,300,900]:

        fit_score_time(clf, X_train[:i], y_train[:i], X_val, y_val, prnt=True, oob=False)

    print("**************************************************************")

end = time.time()

print("Total time needed:", end - start)
# WARNING: This code took about 74 seconds on my machine

knn = KNeighborsClassifier()

sgd = make_pipeline(StandardScaler(),SGDClassifier(random_state=43)) # SGD needs all features to have a similar scale (which we have), so we do not try scaling yet

x_trees = ExtraTreesClassifier(random_state=43)

svc_rbf = make_pipeline(StandardScaler(),SVC(kernel='rbf', random_state=43)) # kernel='rbf' is the default

lin_svc = make_pipeline(StandardScaler(),LinearSVC(random_state=43))

grad_boost = GradientBoostingClassifier(random_state=43)



classifiers = [knn, sgd,x_trees,svc_rbf,lin_svc]



start = time.time()

for clf in classifiers:

    print(clf.__class__.__name__)

    for i in [2700]:

        fit_score_time(clf, X_train[:i], y_train[:i], X_val, y_val, prnt=True, oob=False)

    print("**************************************************************")

end = time.time()

print("Total time needed:", end - start)
# the functions that will do the cropping

def crop_inst(X_inst, tr_im_size=[28,28], crop_left=3, crop_right=3, crop_top=3, crop_bottom=3):

    """crop_data a data instance into its image_size, crops the given pixels from the border

    and transforms the new image back into a row array"""

    X_im = X_inst.reshape(tr_im_size)

    X_im = X_im[crop_top:-crop_bottom, crop_left:-crop_right] # do the cropping

    return X_im.reshape((tr_im_size[0]-crop_top-crop_bottom)*(tr_im_size[1]-crop_left-crop_right))

    



def crop_data(X_train, tr_im_size=[28,28], crop_left=3, crop_right=3, crop_top=3, crop_bottom=3):

    """crop_data turns every data instance into its image_size, crops the given pixels from the border

    and transforms the new image back into a row array, returning thus a new data set"""

    return np.apply_along_axis(crop_inst,1,X_train,tr_im_size=tr_im_size,crop_left=crop_left,

                               crop_right=crop_right, crop_top=crop_top, crop_bottom=crop_bottom)
# crop_inst should work:

plot_digit(crop_inst(X_train[43]), size=22)
# crop_data should work and not take too long

start = time.time()

X_crp = crop_data(X_train)

end = time.time()

print("Time needed to crop all instances:", end-start)

plot_digit(X_crp[43], size=22)
# lets see how a random forest performs on the cropped set

rdf_clf = RandomForestClassifier(random_state=43, oob_score=True, n_jobs=-1)



fit_score_time(rdf_clf, X_crp, y_train_label, crop_data(X_val), y_val_label, prnt=True, oob=True) # took 50 seconds, score 0.987
start = time.time()

for clf in classifiers: # reminder: classifiers = [knn,sgd,x_trees,svc_rbf,lin_svc]

    print(clf.__class__.__name__)

    fit_score_time(clf, X_crp[:2700], y_train[:2700], crop_data(X_val), y_val, prnt=True, oob=False)

    print("**************************************************************")

    

end = time.time()

print("Total time needed:", end - start)
# takes 235 seconds on my machine

if False: # to prevent the kernel to compute again

    start = time.time()

    print("Training instances are not cropped:")

    for clf in [sgd,x_trees]:

        print(clf.__class__.__name__)

        fit_score_time(clf, X_train, y_train, X_val, y_val, prnt=True, oob=False)

        print("**************************************************************")



    print("Training instances are cropped:")

    for clf in [sgd,x_trees]:

        print(clf.__class__.__name__)

        fit_score_time(clf, X_crp, y_train, crop_data(X_val), y_val, prnt=True, oob=False)

        print("**************************************************************")

    end = time.time()

    print("Total time needed:", end - start)
# the function is really easy and should be very fast

def stratify_data(X_train, threshold=230):

    return np.where(X_train>threshold/255, 1, 0).astype(np.bool)
# Okay, lets apply

X_strat = stratify_data(X_train)
# no error yet, so lets look if it works:

plt.figure(figsize=(9,9))

plot_digits(X_strat[:100], images_per_row=10, size=28)
# threshold set to 130

plt.figure(figsize=(9,9))

plot_digits(stratify_data(X_train, threshold=130)[:100], images_per_row=10, size=28)
# threshold set to 0

plt.figure(figsize=(9,9))

plot_digits(stratify_data(X_train, threshold=0)[:100], images_per_row=10, size=28)
def stratify_data_soft(X_train, threshold=130):

    return np.where(X_train>threshold/255, X_train, 0) # /255 added because data pixels are assumed to be in [0,1]
# stratify_data_soft with threshold=130:

plt.figure(figsize=(9,9))

plot_digits(stratify_data_soft(X_train, threshold=130)[:100], images_per_row=10, size=28)
# reshape the data to 28x28 image

some_digit = X_train[42]

some_image = some_digit.reshape(28,28)



# resize the image to 7x7

from skimage.transform import resize

some_scaled_image = resize(some_image,(7,7))



plt.imshow(some_scaled_image, cmap = 'gray') # should be a 9
# Well, this will certainly be too small to train any digit recognizer (but maybe, you can try!)

# reshape the data to 28x28 image

some_digit = X_train[41]

some_image = some_digit.reshape(28,28)



# resize the image to 7x7

from skimage.transform import resize

some_scaled_image = resize(some_image,(14,14))



plt.imshow(some_scaled_image, cmap = 'gray') # should be a nine
# this seems reasonable or at least manageable. lets define the functions (note that the

# mode of the resizer and anti_aliasing are hyperparameter as well. we won't dive in this



def instance_resizer(instance, orig_im_size=(28,28), new_im_size=(14,14), mode='reflect', anti_aliasing=True):

    """takes a vector, interprets it as a picture, resizes it and returns the new image as a vector"""

    image_orig = instance.reshape(orig_im_size)

    image_new = resize(image_orig, new_im_size, mode=mode)

    inst_transformed = image_new.reshape(new_im_size[0]*new_im_size[1])

    return inst_transformed



def resize_train(X_train, orig_im_size=(28,28), new_im_size=(14,14), mode='reflect',anti_aliasing=True):

    """applies instance_resizer to every instance"""

    return np.apply_along_axis(instance_resizer,1,X_train,orig_im_size=orig_im_size,new_im_size=new_im_size,

                               mode=mode, anti_aliasing=anti_aliasing)
# apply on some digit

plt.imshow(instance_resizer(X_train[41]).reshape(14,14), cmap = 'gray')
# looks fine. does it work on X_train? how long will it take?

start = time.time()

X_res = resize_train(X_train)

end = time.time()

print("Time needed for resizing:", end - start) # about 44 seconds

plt.imshow(X_res[41].reshape(14,14), cmap = 'gray')
# looks nice. lets plot some of them:

plt.figure(figsize=(9,9))

plot_digits(X_res[:100], images_per_row=10, size=14)
plot_digit(X_train[54])
# whats roughly the value for the pollution?

print(X_train[54].reshape(28,28)[5,22])

print(X_train[54].reshape(28,28)[4,22])
# here are the functions:

def pollute_instance(instance, im_size=(28,28), n_poll_pic=1, random_state = None):

    if not random_state is None:

        np.random.seed(random_state)

    inst = instance.copy() # otherwise, we change the original data

    image = inst.reshape(im_size)

    # set rectangle with pollution

    for _ in range(n_poll_pic):

        a, b = np.random.randint(0,im_size[0]), 1+int(4*np.random.uniform())

        c, d = np.random.randint(0,im_size[1]), 1+int(4*np.random.uniform())

        # start polluting

        for i in range(a,a+b):

            for j in range(c,c+d):

                try:

                    image[i,j] = np.random.randint(50,200)/255 # /255 added afterwards because pixels now assumed to be in [0,1]

                except IndexError:

                    pass

    inst_transformed = image.reshape(im_size[0]*im_size[1])



    return inst_transformed

    



def pollute_training(X_train, im_size=(28,28), n_poll_pic=1, random_state = None):

    # usually, we would have used the following code, but it does not work with random_state set because in each

    # instance the same pixels are polluted

    if random_state is None:

        return np.apply_along_axis(pollute_instance,1,X_train,im_size=im_size,

                                   n_poll_pic=n_poll_pic, random_state=random_state)

    else:

        X_pol = X_train.copy()

        for i in range(X_train.shape[0]):

            X_pol[i] = pollute_instance(X_train[i],im_size=im_size, n_poll_pic=n_poll_pic,

                                        random_state=random_state+i)

        return X_pol
# lets try:

start = time.time()

X_pol = pollute_training(X_train, random_state=43)

end = time.time()

print("Time needed for polluting training data:", end - start) # is about 5 seconds, so quick enough

plt.figure(figsize=(10,10))

plot_digits(X_pol[:100], images_per_row=10, size=28)
# nice. lets test two polluted areas:

plt.figure(figsize=(10,10))

plot_digits(pollute_training(X_train[:100], n_poll_pic=2), images_per_row=10, size=28)
# shift_instance and rand_shift_training are adapted versions from Aurelien Gerons book chapter 3 additional material

# see his notebooks

# to avoid combinatorial explosion, we use random shifting and rotating



def shift_instance(instance, dx, dy, new=0, im_size=(28,28)):

    return shift(instance.reshape(im_size), [dy, dx], cval=new).reshape(im_size[0]*im_size[1])



def rand_shift_training(X_train, y_train, dx_max=2, dy_max=2, n_shifts = 4, new=0,

                        im_size=(28,28), random_state=None, only_axes=True, do_shuffle=True):

    

    if not random_state is None:

        np.random.seed(random_state)

    

    # collect all possibilities to move an image with given dx_max, dy_max

    moving_vectors = []

    if only_axes:

        for i in range(1,dx_max+1): # if i=0 or j=0, we will get double entries. we will remove them below

            moving_vectors.append((i,0))

            moving_vectors.append((-i,0))

        for i in range(1,dy_max+1):

            moving_vectors.append((0,i))

            moving_vectors.append((0,-i))

    else:

        for i in range(0,dx_max+1):

            for j in range(0,dy_max+1): # if i=0 or j=0, we will get double entries. we will remove them below

                moving_vectors.append((i,j))

                moving_vectors.append((-i,j))

                moving_vectors.append((i,-j))

                moving_vectors.append((-i,-j))

    

    moving_vectors = list(set(moving_vectors)) # removes double entries from moving_vectors



    # we do not want to double images, so remove (0,0) from moving_vectors

    try:

        moving_vectors.remove((0,0))

    except ValueError:

        if not only_axes: # because there should be no (0,0) inside if only_axes is True

            print("No (0,0) could be removed from moving_vectors. Please check the code.")

    

    

    X_train_expanded = []

    y_train_expanded = []

    

    # move image by moving_vectors and add to X_train_expanded

    for i in range(X_train.shape[0]):

        try:

            for dx, dy in random.sample(moving_vectors, n_shifts):

                shifted_image = shift_instance(X_train[i], dx=dx, dy=dy, new=0, im_size=im_size)

                X_train_expanded.append(shifted_image)

                y_train_expanded.append(y_train[i])

        except ValueError:

            print("Using maximal possible shifts. This might be smaller thatn n_shifts.")

            for dx, dy in moving_vectors:

                shifted_image = shift_instance(X_train[i], dx=dx, dy=dy, new=0, im_size=im_size)

                X_train_expanded.append(shifted_image)

                y_train_expanded.append(y_train[i])



    X_train_expanded = np.array(X_train_expanded, dtype=np.uint8)

    y_train_expanded = np.array(y_train_expanded, dtype=np.uint8)

    

    # shuffle the new shifted images to avoid classifiers see the same number for too many times

    if do_shuffle:

        X_train_expanded, y_train_expanded = shuffle(X_train_expanded, y_train_expanded, random_state=random_state)

    return X_train_expanded, y_train_expanded



def rand_rotate_instance(instance, angle=10, new=0, im_size=(28,28)):

    return rotate(instance.reshape(im_size), angle=angle, cval=new, reshape=False).reshape(im_size[0]*im_size[1])



def rand_rotate_training(X_train, y_train, angle_range=(5,10), im_size=(28,28),

                        sign=True, do_shuffle=True, random_state=None):

    

    if not random_state is None:

        np.random.seed(random_state)

    

    X_rot = X_train.copy()

    for i in range(X_train.shape[0]): # we need this loop to not always get the same rotation

        angle = np.random.randint(angle_range[0],angle_range[1])

        X_rot[i] = rand_rotate_instance(X_train[i],angle=angle,im_size=im_size)

    

    if sign:

        X_rot_sign = X_train.copy()

        for i in range(X_train.shape[0]): # we need this loop to not always get the same rotation

            angle = np.random.randint(angle_range[0],angle_range[1])

            X_rot_sign[i] = rand_rotate_instance(X_train[i],angle=-angle,im_size=im_size)

        

        # shuffle the new shifted images to avoid classifiers see the same number for too many times

        if do_shuffle:

            X_train_expanded, y_train_expanded = shuffle(np.concatenate((X_rot, X_rot_sign)),

                                                         np.concatenate((y_train, y_train)),

                                                         random_state=random_state)

        return X_train_expanded, y_train_expanded    

    else:

        return X_rot, y_train
class ImageTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, resize=True, pollute=True, crop=True, stratify=True, n_poll_pic=1, random_state=None): # pass only whether to apply some transformations and random state

        self.resize = resize

        self.crop = crop

        self.stratify = stratify

        self.pollute = pollute

        self.n_poll_pic = n_poll_pic

        self.random_state = random_state

    def fit(self, X, y=None):

        return self  # nothing else to do

    def transform(self, X): 

        if self.pollute:

            X = pollute_training(X,n_poll_pic = self.n_poll_pic, random_state=self.random_state)

        if self.resize:

            if X.shape[1] != 14*14: # only resize if pictures are not already resized to (14,14)

                X = resize_train(X)

        if self.crop:

            X = crop_data(X, tr_im_size=[14,14], crop_left=1, crop_right=1, crop_top=1, crop_bottom=1)

        if self.stratify:

            stratify_data(X, threshold=0)

        return X
def augment_training(X_train, y_train,dx_max=2,dy_max=2,n_shifts = 4,angle_range=(5,10),im_size=(28,28),

                     only_axes=False, do_shuffle=True, sign=True, random_state=None):

    X_shift, y_shift = rand_shift_training(X_train, y_train, dx_max=dx_max, dy_max=dy_max, n_shifts = n_shifts, 

                        im_size=im_size, random_state=random_state, only_axes=only_axes, do_shuffle=do_shuffle)

    X_rot, y_rot = rand_rotate_training(X_train, y_train, angle_range=angle_range, im_size=im_size,

                        sign=sign, do_shuffle=do_shuffle, random_state=random_state)

    X_train = np.concatenate((X_train,X_shift, X_rot))

    y_train = np.concatenate((y_train, y_shift, y_rot))

    return X_train, y_train
if False: # WARNING:took 2457 seconds on my machine

    knn = KNeighborsClassifier()

    sgd = make_pipeline(StandardScaler(),SGDClassifier(random_state=43)) # SGD needs all features to have a similar scale (which we have), so we do not try scaling yet

    x_trees = ExtraTreesClassifier(random_state=43)

    svc_rbf = make_pipeline(StandardScaler(),SVC(kernel='rbf', random_state=43)) #  kernel='rbf' is the default in sklearn

    svc_nyst_rbf = make_pipeline(StandardScaler(),Nystroem(kernel='rbf', random_state=43),LinearSVC())

    grad_boost = GradientBoostingClassifier(random_state=43)



    # now resize True for predicting

    im_transf = ImageTransformer(resize=True, pollute=True, crop=True, stratify=True,random_state=43)

    im_transf2 = ImageTransformer(resize=False, pollute=False, crop=True, stratify=True,random_state=43)



    classifiers = [knn, sgd, x_trees, svc_rbf, svc_nyst_rbf]

    params = [{'crop': [True,False], 'stratify': [True,False]}]



    X_val_res = resize_train(X_val) # so to not always resize again in the loop; we do not want pollute validating set



    start = time.time()

    for pol in [True, False]:

        im_transf.set_params(**{'resize': True, 'pollute': pol, 'crop': False, 'stratify': False,'random_state': 43})

        X = im_transf.transform(X_train)

        for dic in list(ParameterGrid(params)):

            im_transf2.set_params(**dic)

            print("Pollution is set to", pol)

            print("Now apply ImageTransformer with", im_transf2.get_params())

            for clf in classifiers: # reminder: classifiers = [knn,sgd,x_trees,svc_rbf,lin_svc]

                print(clf.__class__.__name__)

                fit_score_time(clf, im_transf2.transform(X), y_train, im_transf2.transform(X_val_res), y_val, prnt=True, oob=False)

                print("**************************************************************")



    end = time.time()

    print("Total time needed:", end - start)
im_transf = ImageTransformer(resize=True, pollute=True, crop=True, stratify=False, n_poll_pic=10, random_state=43)

plt.figure(figsize=(10,10))

plot_digits(im_transf.transform(X_train[:100]), images_per_row=10, size=12)
# thats the training data we work with

im_transf = ImageTransformer(resize=True, pollute=False, crop=False, stratify=False,random_state=43)

X = im_transf.transform(X_train)

X_val_res = im_transf.transform(X_val)
clf = KNeighborsClassifier(n_jobs=-1)

clf_params = [{'n_neighbors': [2,4,8], 'weights': ['uniform', 'distance']}]



for param in list(ParameterGrid(clf_params)): # reminder: classifiers = [knn,sgd,x_trees,svc_rbf,lin_svc]

    print(clf.__class__.__name__)

    clf.set_params(**param)

    print(param)

    fit_score_time(clf, X[:10000], y_train[:10000], X_val_res, y_val, prnt=True, oob=False)

    print("**************************************************************")
clf = ExtraTreesClassifier(random_state=43, n_jobs=-1, oob_score = True, bootstrap=True)

clf_params = [{'max_depth': [3,9,14], 'min_samples_split': [50, 100,500]}]



for param in list(ParameterGrid(clf_params)): # reminder: classifiers = [knn,sgd,x_trees,svc_rbf,lin_svc]

    print(clf.__class__.__name__)

    clf.set_params(**param)

    print(param)

    fit_score_time(clf, X, y_train_label, X_val_res, y_val_label, prnt=True, oob=False)

    print("oob score is:", clf.oob_score_)

    print("**************************************************************")
# are the validation labels correct?

print(y_val[:20])

print(y_val_label[:20])
# so lets put them together into a voting team

svc_nyst = Pipeline([('std_scaler',StandardScaler()),

                ('nystroem',Nystroem(kernel='rbf', gamma=0.0034, random_state=43)),

                ('lin_svc',LinearSVC(C=5, dual=False, random_state=43))])

svc_rbf = Pipeline([('std_scaler',StandardScaler()),('rbf',SVC(kernel='rbf', C=5, gamma=0.005, random_state=43))])

rnd_for = RandomForestClassifier(random_state=43, n_jobs=-1)

x_trees = ExtraTreesClassifier(random_state=43, n_jobs=-1)

knn = KNeighborsClassifier(n_neighbors=4, weights='distance')

sgd = make_pipeline(StandardScaler(),SGDClassifier(alpha= 0.0005, loss='hinge', penalty='l2', n_jobs=-1,

                                                        random_state=43))



vote_fast = VotingClassifier([('nystroem', svc_nyst), ('sgd',sgd)])

vote_slow = VotingClassifier([('rbf', svc_rbf), ('x_trees',x_trees), ('rnd_for',rnd_for), ('knn',knn)])
X_aug, y_aug = augment_training(resize_train(X_train),y_train,

                                dx_max=2,dy_max=2, n_shifts=4, im_size=(14,14),random_state=43)
start = time.time()

vote_fast.fit(X_aug,y_aug)

end = time.time()

print("Time fitting the fast voters:", end-start) # 132 seconds
start = time.time()

vote_slow.fit(X_res,y_train)

end = time.time()

print("Time fitting the slow voters:", end-start) # 150 seconds
X_test_res = resize_train(X_test)
print("Score on the validation set:",accuracy_score(y_val, vote_slow.predict(X_val_res)))

print("Score on the test set:",accuracy_score(y_test, vote_slow.predict(X_test_res)))
for estimator in vote_slow.estimators_: # svc_rbf, extra_trees, random_forest, knn

    print("Score of", estimator.__class__.__name__, "on the validation set:", estimator.score(X_val_res, y_val))

    print("Score of", estimator.__class__.__name__, "on the test set:", estimator.score(X_test_res, y_test))
print("Score on the validation set:",accuracy_score(y_val, vote_fast.predict(X_val_res)))

print("Score on the test set:",accuracy_score(y_test, vote_fast.predict(X_test_res)))
for estimator in vote_fast.estimators_: # nymstroem, sgd

    print("Score of", estimator.__class__.__name__, "on the validation set:", estimator.score(X_val_res, y_val))

    print("Score of", estimator.__class__.__name__, "on the test set:", estimator.score(X_test_res, y_test))
X_eval = df_test.to_numpy()

X_eval_res = resize_train(X_eval)
# this takes long because knn needs some time to predict 28000 instances

#y_eval = vote_slow.predict(X_eval_res) t
#df = pd.DataFrame(y_eval, columns=['Label'])

#df.index = np.arange(1, len(df)+1)

#df.to_csv(../output/submission.csv, index_label='ImageId') # scored 0.97342
from __future__ import print_function, division

import numpy as np

import matplotlib.pyplot as plt

import os # for operating system commands like dealing with paths

DATA_PATH = '../input' # where are the test.csv and train.csv files located

test_data_path = os.path.join(DATA_PATH, 'test.csv')

train_data_path = os.path.join(DATA_PATH, 'train.csv')
%%time

# this takes around 30 seconds so be patient

train_data = np.loadtxt(train_data_path, delimiter = ',', skiprows = 1)

numb_id = train_data[:,0] # just the number id

numb_vec = train_data[:,1:] # the array of the images
print('Input Data:', train_data.shape)

print('Number ID:', numb_id.shape)

print('Number Vector:', numb_vec.shape)
numb_image = numb_vec.reshape(-1, 28, 28)

print('Number Image', numb_image.shape)
%matplotlib inline

fig, ax1 = plt.subplots(1,1)

ax1.matshow(numb_image[0], cmap = 'gray')

ax1.set_title('Current Digit {}'.format(numb_id[0]))

ax1.axis('off')
from skimage.measure import label # connected component labeling

def seg_and_label(in_image):

    norm_image = (in_image - in_image.mean())/in_image.std()

    return (norm_image>0.5).astype(np.uint8)
fig, (ax1,ax2) = plt.subplots(1,2)

ax1.matshow(numb_image[0], cmap = 'gray')

ax1.set_title('Current Digit {}'.format(numb_id[0]))

ax1.axis('off')

ax2.matshow(seg_and_label(numb_image[0]),cmap='gist_earth')

ax2.set_title('Segmented and Labeled')

ax2.axis('off')
from skimage.measure import regionprops

def shape_analysis(in_label):

    try:

        mean_anisotropy=np.mean([(freg.major_axis_length-freg.minor_axis_length)/freg.minor_axis_length 

                                 for freg in regionprops(in_label)])

    except ZeroDivisionError:

        mean_anisotropy = 0

    return dict(

           total_area=np.sum([freg.area for freg in regionprops(in_label)]),

    total_perimeter=np.sum([freg.perimeter for freg in regionprops(in_label)]),

    mean_anisotropy=mean_anisotropy,

        mean_orientation=np.mean([freg.orientation for freg in regionprops(in_label)])

               )



shape_analysis(seg_and_label(numb_image[0]))
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier



digit_examples = KNeighborsClassifier(1) 



class PipeStep(object):

    """

    Wrapper for turning functions into pipeline transforms (no-fitting)

    """

    def __init__(self, step_func):

        self._step_func=step_func

    def fit(self,*args):

        return self

    

    def transform(self,X):

        return self._step_func(X)





norm_step = PipeStep(lambda in_image: (in_image - in_image.mean())/in_image.std())

thresh_step = PipeStep(lambda img: (img>0.5).astype(np.uint8))

shape_step = PipeStep(lambda img_list: [shape_analysis(img) for img in img_list])

feature_step = PipeStep(lambda shape_list: np.vstack([np.array(list(shape_dict.values())).reshape(1,-1) 

                        for shape_dict in shape_list]))





shape_knn_pipeline = Pipeline([

    ('norm_image', norm_step),

    ('threshold', thresh_step),

    ('shape_analysis', shape_step),

    ('feature', feature_step),

    ('KNN', KNeighborsClassifier(1)) # use just the 1st one (more is often better)

                              ])
%%time

train_idx=np.random.permutation(range(len(numb_id)))[:1000]

print(len(train_idx),'examples')

shape_knn_pipeline.fit(numb_image[train_idx],numb_id[train_idx])
rand_digit = np.random.choice(range(len(numb_image))) # just picks a random digit

guess_digit=shape_knn_pipeline.predict(np.expand_dims(numb_image[rand_digit],0))

print('Guessed {}, actual result was {}'.format(guess_digit, numb_id[rand_digit]))

# show the results

fig, (ax_img) = plt.subplots(1,1, figsize = (5, 5))

ax_img.imshow(numb_image[rand_digit], cmap = 'gray', interpolation = 'none')

ax_img.set_title('Guessed {}, actual result was {}'.format(guess_digit, numb_id[rand_digit]))
test_vec = np.loadtxt(test_data_path, delimiter = ',', skiprows = 1)

print('Test Vec', test_vec.shape)
test_image = test_vec.reshape(-1, 28, 28)

print('Test Image', test_image.shape)
%%time

guess_test_data = shape_knn_pipeline.predict(test_image)
with open('submission.csv', 'w') as out_file:

    out_file.write('ImageId,Label\n')

    for img_id, guess_label in enumerate(guess_test_data):

        out_file.write('%d,%d\n' % (img_id+1, guess_label))
from sklearn.pipeline import Pipeline



class ShapeAnalysisVector(object):

    def fit(self,*args):

        return self

    

    def transform(self,X):

        return self._apply(X)

    

    @staticmethod

    def _apply(in_img_vec):

        out_vec=np.vstack([full_analysis(x_img).reshape(1,-1) for x_img in in_img_vec.reshape(-1,28,28)])

        return out_vec



shape_knn_pipeline = Pipeline([('shape_analysis', ShapeAnalysisVector()), 

                               ('KNN', KNeighborsClassifier(1))])
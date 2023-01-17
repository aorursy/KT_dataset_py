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
from scipy.ndimage.morphology import distance_transform_edt as distmap

def seg_image(in_image):

    norm_image = (in_image - in_image.mean())/in_image.std()

    return (norm_image>0.5).astype(np.uint8)
fig, (ax1,ax2, ax3) = plt.subplots(1,3)

ax1.matshow(numb_image[0], cmap = 'gray')

ax1.set_title('Current Digit {}'.format(numb_id[0]))

ax1.axis('off')

ax2.matshow(seg_image(numb_image[0]),cmap='gist_earth')

ax2.set_title('Segmented')

ax2.axis('off')



ax3.matshow(distmap(seg_image(numb_image[0])==0),cmap='magma')

ax3.set_title('Distance Map')

ax3.axis('off')
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier



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

dist_step = PipeStep(lambda img_list: [distmap(img==0) for img in img_list])

flatten_step = PipeStep(lambda img_list: [img.ravel() for img in img_list])





dist_rf_pipeline = Pipeline([

    ('norm_image', norm_step),

    ('threshold', thresh_step),

    ('Distance Map', dist_step),

    ('Flatten Image', flatten_step),

    ('RF', RandomForestClassifier())

                              ])
%%time

train_idx=np.random.permutation(range(len(numb_id)))[:25000]

print(len(train_idx),'examples')

dist_rf_pipeline.fit(numb_image[train_idx],numb_id[train_idx])
rand_digit = np.random.choice(range(len(numb_image))) # just picks a random digit

guess_digit=dist_rf_pipeline.predict(np.expand_dims(numb_image[rand_digit],0))

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

guess_test_data = dist_rf_pipeline.predict(test_image)
with open('submission.csv', 'w') as out_file:

    out_file.write('ImageId,Label\n')

    for img_id, guess_label in enumerate(guess_test_data):

        out_file.write('%d,%d\n' % (img_id+1, guess_label))
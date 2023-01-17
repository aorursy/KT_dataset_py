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
# make it into a feature vector

def full_analysis(in_img):

    return np.array(list(shape_analysis(seg_and_label(in_img)).values()))


from sklearn.neighbors import KNeighborsClassifier

digit_examples = KNeighborsClassifier(1) # use just the 1st one (more is often better)

knn_train_vec=[]

knn_train_label=[]

for cur_digit in np.unique(numb_id):

    # find the first example

    digit_matches = np.where(numb_id == cur_digit)[0]

    for cur_idx in np.random.choice(digit_matches,size=5,replace=True):

        knn_train_vec+=[full_analysis(numb_image[cur_idx]).reshape(1,-1)]

        knn_train_label+=[cur_digit]
digit_examples.fit(np.vstack(knn_train_vec), knn_train_label)
rand_digit = np.random.choice(range(len(numb_image))) # just picks a random digit



guess_digit = digit_examples.predict(

    full_analysis(numb_image[rand_digit]).reshape(1,-1)

)

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

guess_test_data = [digit_examples.predict(full_analysis(c_img).reshape(1,-1))[0] for c_img in test_image]
with open('submission.csv', 'w') as out_file:

    out_file.write('ImageId,Label\n')

    for img_id, guess_label in enumerate(guess_test_data):

        out_file.write('%d,%d\n' % (img_id+1, guess_label))
from sklearn.pipeline import Pipeline



class ShapeAnalysisVector(object):

    def fit(self,X,y):

        return self

    def transform(self,X):

        return np.vstack([full_analysis(x_img).reshape(1,-1) for x_img in X.reshape(-1,28,28)])
shape_knn_pipeline = Pipeline([('shape_analysis', ShapeAnalysisVector()), 

                               ('KNN', KNeighborsClassifier(1))])
shape_knn_pipeline.fit(numb_vec[:10],numb_id[:10])
shape_knn_pipeline.predict(numb_vec[:10])
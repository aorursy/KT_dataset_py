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
digit_examples = {} # we use a dictionary to store the results

digit_features = {} # we use a dictionary to store the results



for cur_digit in np.unique(numb_id):

    # find the first example

    digit_matches = np.where(numb_id == cur_digit)

    digit_examples[cur_digit] = numb_image[digit_matches[0][0]]

    digit_features[cur_digit] = full_analysis(numb_image[digit_matches[0][0]])
fig, c_axs = plt.subplots(1, len(digit_examples), figsize = (20, 3))

for c_ax, (c_digit, c_img) in zip(c_axs, digit_examples.items()):

    c_ax.imshow(c_img, cmap = 'gray', interpolation = 'none')

    c_ax.set_title('Digit {}'.format(c_digit))

    c_ax.axis('off')

    print('Digit {}'.format(c_digit))

    print(full_analysis(c_img))
def mae(fvec1,fvec2):

    return np.mean(np.abs(fvec1-fvec2))



def classify_image(example_dict, in_image):

    score_dict = {c_digit: mae(full_analysis(in_image), c_fvec) for c_digit, c_fvec in example_dict.items()}

    best_digit, best_score = sorted(score_dict.items(), 

                                    key = lambda x: x[1])[0] # sort by score and take the first item

    return best_digit, score_dict
rand_digit = np.random.choice(range(len(numb_image))) # just picks a random digit



guess_digit, guess_dict = classify_image(digit_features, numb_image[rand_digit])

print('Guessed {}, actual result was {}'.format(guess_digit, numb_id[rand_digit]))

print('Score for other numbers:', guess_dict)

# show the results

fig, (ax_img, ax_score) = plt.subplots(1,2, figsize = (10, 5))

ax_img.imshow(numb_image[rand_digit], cmap = 'gray', interpolation = 'none')

ax_img.set_title('Guessed {}, actual result was {}'.format(guess_digit, numb_id[rand_digit]))

ax_score.bar(list(guess_dict.keys()), list(guess_dict.values()))

ax_score.set_xlabel('Digit')

ax_score.set_ylabel('MSE')

ax_score.set_title('MAE for each digit feature vector')
test_vec = np.loadtxt(test_data_path, delimiter = ',', skiprows = 1)

print('Test Vec', test_vec.shape)

test_image = test_vec.reshape(-1, 28, 28)

print('Test Image', test_image.shape)
%%time

guess_test_data = [classify_image(digit_features, c_img)[0] for c_img in test_image]
with open('submission.csv', 'w') as out_file:

    out_file.write('ImageId,Label\n')

    for img_id, guess_label in enumerate(guess_test_data):

        out_file.write('%d,%d\n' % (img_id+1, guess_label))
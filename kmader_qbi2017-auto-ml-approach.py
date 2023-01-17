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
from tpot import TPOTClassifier

auto_classifier = TPOTClassifier(generations=1, population_size=5, verbosity=2)
%%time

np.random.seed(1234)

test_idx = np.random.choice(range(len(numb_vec)), 8000) # since the whole dataset takes up too much memory

auto_classifier.fit(numb_vec[test_idx], numb_id[test_idx])
rand_digit = np.random.choice(range(len(numb_image))) # just picks a random digit

rand_digit_vec = numb_image[rand_digit].reshape(1,-1)

guess_digit = auto_classifier.predict(rand_digit_vec)

# show the probabilities for each class

guess_digit_prob = auto_classifier._fitted_pipeline.predict_proba(rand_digit_vec)

guess_dict = dict(enumerate(guess_digit_prob[0]))

print('Guessed {}, actual result was {}'.format(guess_digit, numb_id[rand_digit]))

print('Score for other numbers:', guess_dict)

# show the results

fig, (ax_img, ax_score) = plt.subplots(1,2, figsize = (10, 5))

ax_img.imshow(numb_image[rand_digit], cmap = 'gray', interpolation = 'none')

ax_img.set_title('Guessed {}, actual result was {}'.format(guess_digit, numb_id[rand_digit]))

ax_score.bar(list(guess_dict.keys()), list(guess_dict.values()))

ax_score.set_xlabel('Digit')

ax_score.set_ylabel('Probability')

ax_score.set_title('Probability for each digit')
test_vec = np.loadtxt(test_data_path, delimiter = ',', skiprows = 1)

print('Test Vec', test_vec.shape)

test_image = test_vec.reshape(-1, 28, 28)

print('Test Image', test_image.shape)
%%time

guess_test_data = auto_classifier.predict(test_vec)
with open('submission.csv', 'w') as out_file:

    out_file.write('ImageId,Label\n')

    for img_id, guess_label in enumerate(guess_test_data):

        out_file.write('%d,%d\n' % (img_id+1, guess_label))
#Importing required Dependencies

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams['figure.figsize'] = [16, 10]

plt.rcParams['font.size'] = 16



import os

from tqdm import tqdm # Fancy progress bars



import seaborn as sns

from keras.preprocessing import image

from keras.applications import xception

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix



# Our data files are available in the "../input/" directory.

print(os.listdir("../input"))

# For Kaggle kernel purposes any results we write to the current directory is saved as output.
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
#Loading the Keras Pretrained Model into Kaggle Kernels

#Copying the Keras pretrained models into the cache directories and displaying the pretrained models that we have prepared in our file directory



!ls ../input/keras-pretrained-models/
#Creating the keras cache directories in Kaggle Kernels to load the pretrained models

cache_dir = os.path.expanduser(os.path.join('~', '.keras')) #The Cache directory

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models') #The Models directory

if not os.path.exists(models_dir):

    os.makedirs(models_dir)
#Copying a selection of our pretrained models files onto the keras cache directory for Keras to access

!cp ../input/keras-pretrained-models/xception* ~/.keras/models
#Displaying the pretrained models

!ls ~/.keras/models
!ls ../input/plant-seedlings-classification
#Preparing the dataset for the model

#Defining Y-labels of the classes of the dataset

#Defining the NUM_CLASSES, i.e. total classes in the dataset

CATEGORIES = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen', 'Loose Silky-bent',

             'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

NUM_CATEGORIES = len(CATEGORIES)
SAMPLE_PER_CATEGORY = 200

SEED = 7

data_dir = '../input/plant-seedlings-classification/'

train_dir = os.path.join(data_dir, 'train')

test_dir = os.path.join(data_dir, 'test')

sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))



print(train_dir)

print(test_dir)
sample_submission.head(10)
#Displaying the training data: Note that the training images are organized into sub-folders within the main folder,organized by plant species.



for category in CATEGORIES:

    print('{} {} images'.format(category, len(os.listdir(os.path.join(train_dir, category)))))

    # "Print length of this directory -- an integer output"
#Creating Aggregate Training Sample for CNN

#Traversing over the directories and folders containing the training set data to collate all the image-files

#and their corresponding class index and class_names into training-set collection, and converting into a pandas DataFrame.



#Aggregating the data (i.e.: filenames) and their labels.





train = []

for category_id, category in enumerate(CATEGORIES): #category_id is the integer index corresponding to each class_name

    for file in os.listdir(os.path.join(train_dir, category)): 

        train.append(['train/{}/{}'.format(category, file), category_id, category]) #Renaming the file names and adding to the train list

        

train = pd.DataFrame(train, columns = ['file', 'category_id', 'category']) #Defining a pandas DataFrame over training data

train.head(5) #Print preview of the training DataFrame

train.shape 
#Creating our training set

train = pd.concat([train[train['category'] == c][:SAMPLE_PER_CATEGORY] for c in CATEGORIES])

train = train.sample(frac=1) #Returning a random sample of items from an axis using pandas function with axis defaults to =0



train.index = np.arange(len(train)) #Specifying the DataFrame's index

train.head(5)

train.shape #m decreased because we selected a random sample from the aggregate training-set
#Creating our test set

#Collating all the test examples into a neatly organized pandas DataFrame with appropriate headers



test = []

for file in os.listdir(test_dir):

    test.append(['test/{}'.format(file), file])

test = pd.DataFrame(test, columns=['filepath', 'file'])

test.head(5)

test.shape #We would expect (m, 2) with m being the number of test examples, and 2 being the filepath and file columns
#Reading an Image to an Array

# Image is a keras.preprocessing object containing function for preprocessing images for use in keras / tf models

# Essentially, converting images into their corresponding 3-D numpy arrays

#concating the filepaths, and the function will spit out the image file's array format



def read_img(filepath, size):

    img = image.load_img(os.path.join(data_dir, filepath), target_size = size)

    img = image.img_to_array(img)

    return img
#Loading and Visualizing Sample Images (Training Examples)

# Using matplotlib



fig = plt.figure(1, figsize=(NUM_CATEGORIES, NUM_CATEGORIES)) # Displaying a square matrix with num_categories number of

#images for each category, across all categories

grid = ImageGrid(fig, 111, nrows_ncols=(NUM_CATEGORIES, NUM_CATEGORIES), axes_pad=0.05) #Set-up grid using 'fig'

i = 0 # Initialize counter



#Iterate through the files in the categories

for category_id, category in enumerate(CATEGORIES):

    for filepath in train[train['category'] == category]['file'].values[:NUM_CATEGORIES]:

        ax = grid[i]

        img = read_img(filepath, (224,224)) 

        ax.imshow(img/255.)

        ax.axis('off')

        if i % NUM_CATEGORIES == NUM_CATEGORIES - 1: #Labeling the row-categories

            ax.text(250, 112, filepath.split('/')[1], verticalalignment='center')

        i += 1

plt.show();
#Train-Validation Split

#A bit more sophisticated / randomized method of splitting train-dev than simply picking the split index to be

#len(trainset) * split_percentage



np.random.seed(seed=SEED)

rnd = np.random.random(len(train)) 

train_idx = rnd < 0.8 #Indices in which rnd is <0.8 (which should come out to roughly 80% of the dataset)

valid_idx = rnd >= 0.8

ytr = train.loc[train_idx, 'category_id'].values #pandas function calls

yv = train.loc[valid_idx, 'category_id'].values

len(ytr)

len(yv)
#Run Examples through the Pre-trained Xception Model to Extract Xception Features / Representations:

#Specify parameters:

INPUT_SIZE = 299

POOLING = 'avg'

x_train = np.zeros((len(train), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

# Initialize aggregate trainset object of shape (m_total, height, width, channels)



# Filling the numpy array with image files converted into their image-3D arrays

for i, file in tqdm(enumerate(train['file'])): # tqdm is a progress bar

    img = read_img(file, (INPUT_SIZE, INPUT_SIZE))

    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0)) #Pre-process that into a format for Xception model

    x_train[i] = x #Set the i-th example in our initialized zero-4D-array to the particular example

print('Train Images shape: {} size: {:,}'.format(x_train.shape, x_train.size))
#Spliting X into training and validation



Xtr = x_train[train_idx]

Xv = x_train[valid_idx]

print((Xtr.shape, Xv.shape, ytr.shape, yv.shape)) # Print shapes to confirm dims are correct
# Forward propagation through pre-trained Xception model for feature-extraction

#Defining Xception object based on "off-the-shelf" pre-trained Xception model



xception_bottleneck = xception.Xception(weights='imagenet', include_top=False, pooling=POOLING) 

train_x_bf = xception_bottleneck.predict(Xtr, batch_size=32, verbose=1) #Fwdprop through Xception for feature-extraction

valid_x_bf = xception_bottleneck.predict(Xv, batch_size=32, verbose=1)



#Checking output dims:

print("Xception train bottleneck-features shape: {} size: {:,}".format(train_x_bf.shape, train_x_bf.size))

print("Xception valid bottleneck-features shape: {} size: {:,}".format(valid_x_bf.shape, valid_x_bf.size))
#LogReg Classification on ("using") Resulting Xception-bottleneck Features:

#Defining logistic regression object



logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=SEED)

logreg.fit(train_x_bf, ytr) # We need to fit the classifier to our (X,Y pairs)

valid_probs = logreg.predict_proba(valid_x_bf) # Classification on our dev set -- probabilities of various classes

valid_preds = logreg.predict(valid_x_bf) # Classification on our dev set -- predicted classes
#Finding out the accuracy using accuracy_score is an object we've imported from sk-learn



print("Validation Xception Accuracy: {}".format(accuracy_score(yv, valid_preds)))
#Illustrating the Results: Confusion Matrix

cnf_matrix = confusion_matrix(yv, valid_preds) # Confusion matrix imported from sk-learn

abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']

pd.DataFrame({'class': CATEGORIES, 'abbreviation': abbreviation})
#Plotting the confusion matrix to illustrate correct and incorrect predictions



fig, ax = plt.subplots(1)

ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)

ax.set_xticklabels(abbreviation)

ax.set_yticklabels(abbreviation)

plt.title('Confusion Matrix')

plt.ylabel('True Class')

plt.xlabel('Predicted Class')

fig.savefig('Confusion matrix.png', dpi=300)

plt.show();
#Finalization and Creating the Submission

#Creating the X input objects for the test data



x_test = np.zeros((len(test), INPUT_SIZE, INPUT_SIZE, 3), dtype='float32')

for i, filepath in tqdm(enumerate(test['filepath'])):

    img = read_img(filepath, (INPUT_SIZE, INPUT_SIZE))

    x = xception.preprocess_input(np.expand_dims(img.copy(), axis=0))

    x_test[i] = x

print("Test images dataset shape: {} size: {:,}".format(x_test.shape, x_test.size))
# Running forwardprop on the test set input through Xception to get encoded-feature-representation



test_x_bf = xception_bottleneck.predict(x_test, batch_size=32, verbose=1)

print('Xception test bottleneck features shape: {} size: {:,}'.format(test_x_bf.shape, test_x_bf.size))



#Running encoded-feature-representations through the Logistic-Regression classifier (by sk-learn)



test_preds = logreg.predict(test_x_bf)
#Creating the submission file



test['category_id'] = test_preds

test['species'] = [CATEGORIES[c] for c in  test_preds]

test[['file', 'species']].to_csv('submission_101703017.csv', index=False)
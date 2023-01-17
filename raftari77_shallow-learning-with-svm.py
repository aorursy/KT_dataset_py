# import pandas

import pandas as pd 



# import numpy 

import numpy as np



# what we need to plot

%matplotlib inline

import matplotlib.pyplot as plt



from PIL import Image



# set up the display function

from IPython.display import display



import os

from pathlib import Path



# import the convertor

from skimage.color import rgb2gray



import matplotlib as mpl



# import HOG

from skimage.feature import hog



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA



# import train_test_split from sklearn's model selection module

from sklearn.model_selection import train_test_split 



# import SVC from sklearn's svm module

from sklearn.svm import SVC



# import accuracy_score from sklearn's metrics module

from sklearn.metrics import roc_curve, auc, accuracy_score, classification_report
# Open an image

img = Image.open('../input/intel-image-classification/seg_train/seg_train/buildings/4.jpg')



# The image size

print('The image size is:', img.size)



# Seeing the image

img
# Crop the image

img_cropped = img.crop([25, 25, 75, 75])

display(img_cropped)



# Rotate the image

img_rotated = img.rotate(45)

display(img_rotated)



# Flip the image left to right

img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)

display(img_flipped)
# Trun the image into a numpy array 

img_data = np.array(img)



# Get the shape of the aaray

print('The shape of the array is:', img_data.shape)



# Plot the data with imshow

plt.imshow(img_data)

plt.show()



# Plot the red channel 

plt.imshow(img_data[:,:,0], cmap=plt.cm.Reds_r)

plt.show()



# Plot the green channel

plt.imshow(img_data[:,:,1], cmap=plt.cm.Greens_r)

plt.show()



# Plot the blue channel

plt.imshow(img_data[:,:,2], cmap=plt.cm.Blues_r)

plt.show()
# Resize the image

img_resized = np.array(img.resize((50,50)))

# Plot the data with imshow

plt.imshow(img_resized)

plt.show()



img_resized.shape
# Plot the Kernel Density Estimate

def plot_kde(channel, color):

    data = channel.flatten()

    return pd.Series(data).plot.density(c=color)



# The list of channels

channels = ['r', 'g', 'b']



def plot_rgb(image_data):

    for i, color in enumerate(channels):

        plot_kde(image_data[:,:,i], color)

    plt.show()

    

plot_rgb(img_resized)
path = '../input/intel-image-classification/seg_train/seg_train'

for folder in os.listdir(path):

    print(folder)

    for file in os.listdir(Path(path, folder))[0:3]:

        image = Image.open(Path(path, folder, file))

        image_data = np.array(image.resize((150, 150)))

        plt.subplot(1, 2, 1).imshow(image_data)

        plt.subplot(1, 2, 2)

        plot_rgb(image_data)

# Convert image to grayscale

img_bw = img.convert('L')

display(img_bw)



# Convert to the image to array

img_bw_arr = np.array(img_bw.resize((150,150)))



# Get the shape

print('The shape of the array:', img_bw_arr.shape)



# Plot the array using matplotlib

plt.imshow(img_bw_arr, cmap=plt.cm.gray)

plt.show()



# Plot the kde of the new array

plot_kde(img_bw_arr, 'k')
# Labels as a dictionary

image_labels = { 'buildings':0, 'forest':1, 'glacier':2, 'mountain':3, 'sea':4, 'street':5}



# Labels as a dictionary (inverse)

image_labels_inv = { 0:'buildings', 1:'forest', 2:'glacier', 3:'mountain', 4:'sea', 5:'street'}





# create an empty list

labels = []



# Loop over images

for folder in os.listdir(path):

    for file in os.listdir(Path(path, folder)):

        img = Image.open(Path(path, folder, file))

        #if np.array(img).shape == (150, 150, 3): 

        labels.append([(int(os.path.splitext(file)[0])), image_labels.get(folder)])

        

labels = pd.DataFrame(labels, columns=['id','label'])

labels = labels.set_index('id')

labels.head()
# Convert an image id into the numpy array of the corresponding image

def get_image(row_id):

    filename = '{}.jpg'.format(row_id)

    folder = image_labels_inv.get(labels.loc[row_id, 'label'])

    filepath = Path(path, folder, filename)

    img = Image.open(filepath)

    return np.array(img.resize((150,150)))
#  Data from the tenth id in the sea images

sea_id = labels[labels.label == 4].index[9]



plt.imshow(get_image(sea_id))

plt.show() 
# Number of images per class

for i in range(6):

    print(i, labels[labels.label == i].size)
# the mentioned sea image

sea = get_image(sea_id)



# Get the shape of the sea image

print('The color image has shape: ', sea.shape)



# Convert to greyscale using rgb2gray

gray_sea = rgb2gray(sea)



# Show the greyscale image

plt.imshow(gray_sea, cmap=mpl.cm.gray)





# Get the shape of the gray image

print('The gray image has shape: ', gray_sea.shape)
# Applying HOG on the gray sea

hog_features, hog_image = hog(gray_sea, visualize=True, block_norm='L2-Hys', pixels_per_cell=(16,16))



# Show the HOG image

plt.imshow(hog_image, cmap=mpl.cm.gray)
def create_features(img):

    # Flatten the color image

    color_features = img.flatten()

    # Convert image to grayscale

    gray_image = rgb2gray(img)

    # Get HOG of the gray image

    hog_features = hog(gray_image, block_norm='L2-Hys', pixels_per_cell=(16,16))

    # Combine color features

    flat_features = np.hstack((color_features, hog_features))

    return flat_features

    

sea_features = create_features(sea)



# Get the shape of sea image features

print(sea_features.shape)



image_features = sea_features
def create_feature_matrix(df):

    features_list = []

    

    for img_id in df.index:

        # Load image

        img = get_image(img_id)

        # Get features for image

        image_features = create_features(img).astype(np.float32)

        # Add to the list

        #features_list = np.hstack((features_list, image_features))

        features_list.append(image_features)



    # Convert the list into a matrix

    features_list = np.array(features_list)

    return features_list



# Run the defined function

feature_matrix = create_feature_matrix(labels)
# Get the shape of feature matrix

print('The shape of the feature matrix is:', feature_matrix.shape)
# Initiate StandardScaler

ss = StandardScaler()

# Apply the scaler

feature_matrix = ss.fit_transform(feature_matrix)
pca = PCA(n_components=7000)

# use fit_transform to run PCA on our standardized matrix

feature_matrix = pca.fit_transform(feature_matrix)

# look at new shape

print('PCA matrix shape is: ', feature_matrix.shape)
y = labels.label.values

X_train, X_test, y_train, y_test = train_test_split(feature_matrix, y, test_size=.3)



# look at the distrubution of labels in the train set

pd.Series(y_train).value_counts()
# define support vector classifier

svm = SVC(kernel='linear', probability=True)



# fit model

svm.fit(X_train, y_train)
# generate predictions

y_pred = svm.predict(X_test)



# calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print('Model accuracy is: ', accuracy)
classification_report(y_test, y_pred)
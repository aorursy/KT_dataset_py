%load_ext autoreload
%autoreload 2

%matplotlib inline

from fastai.imports import *
from PIL import Image
from imageio import imread

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
import cv2
import matplotlib.pyplot as plt

from sklearn import metrics

import seaborn as sns
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH = '../input'
!ls {PATH}
# upload training data
df_raw = pd.read_csv(f'{PATH}/train.csv')
# load images into an array
def loadem(ids):
    """Load all channels of the given ids into a DataFrame
    """
    images = np.empty((len(ids), 4, 512, 512))
    for index, id in enumerate(ids):
        images[index][0] = imread(f'{PATH}/train/'+id+'_red.png')    # red
        images[index][1] = imread(f'{PATH}/train/'+id+'_green.png')  # green
        images[index][2] = imread(f'{PATH}/train/'+id+'_blue.png')   # blue
        images[index][3] = imread(f'{PATH}/train/'+id+'_yellow.png') # yellow
    return images

# load all channels of an ID to display it
def load_image(id):
    """Load all channels of the given 
    """
    red    = imread(f'{PATH}/train/'+id+'_red.png')
    green  = imread(f'{PATH}/train/'+id+'_green.png')
    blue   = imread(f'{PATH}/train/'+id+'_blue.png')
    yellow = imread(f'{PATH}/train/'+id+'_yellow.png')
    data = np.reshape(yellow, newshape=(512, 512, 1))
    #rgb = Image.merge("RGB",(red, green, blue ))
    #array = np.array(rgb) + data/2
    # add the yellow to the image
    return [red, green, blue, yellow]

# plot the different channels
def plotem(red, green, blue, yellow):
    """Plot the different channels in a row
    """
    images = [red, green, blue, yellow]
    titles = ['Reds', 'Greens', 'Blues', 'Oranges']
    plt.figure(figsize=(15, 10))
    for i in range(len(images)):
        # plot 
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], cmap=titles[i])
        plt.title(titles[i])
    
    plt.tight_layout()
    plt.show()

#rgb, red, green, blue, yellow = combine('0032a07e-bba9-11e8-b2ba-ac1f6b6435d0')
#plot(rgb, red, green, blue, yellow)
#ids = df_raw.Id.head(6)
#rgbs = ids.map(lambda id: combine(id))
#display(rgbs[1])
#for id in ids:
#    rgb = combine(id)
#    display(rgb)
    #plt.figure()
    #plt.imshow(rgb)

# for every label plot one image with all it's channels
labels = ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center", "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", "Golgi apparatus", "Peroxisomes", "Endosomes", "Lysosomes", "Intermediate filaments", "Actin filaments", "Focal adhesion sites", "Microtubules", "Microtubule ends", "Cytokinetic bridge", "Mitotic spindle", "Microtubule organizing center", "Centrosome", "Lipid droplets", "Plasma membrane", "Cell junctions", "Mitochondria", "Aggresome", "Cytosol", "Cytoplasmic bodies", "Rods & rings"]

for index in range(len(labels)):
    print(labels[index])
    target = str(index)
    df_raw[df_raw['Target']==target]['Id']
    ids = df_raw[df_raw['Target']==target]['Id'].head(1)
    for imgid in ids:
        red, green, blue, yellow = load_image(imgid)
        plotem(red, green, blue, yellow)
        
# split the string into list of indexes 
df_raw['Target'] = df_raw['Target'].str.split()

counts = pd.DataFrame(df_raw['Target'].map(lambda x: len(x)), dtype=np.int8)
counts['count'] = 1
ax = counts.groupby(['Target']).agg(['count']).plot(marker='.', title='Distribution of number of proteins')
ax.set_xlabel("number of proteins")
ax.set_ylabel("count")
ax.set_label(None)
# generate one hot encoded columns for each protein
for index in range(len(labels)):
    df_raw[labels[index]] = df_raw.apply(lambda _: int(str(index) in _.Target), axis=1)
# drop the original Target column
df_raw = df_raw.drop('Target', axis=1)
# plot the distribution of proteins
proteins = df_raw.sum(axis=0).drop('Id', axis=0)
proteins = proteins.sort_values(ascending=False)
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 10))
# Plot the total count of appearance for each protein
sns.barplot(proteins.values, proteins.keys())
    
# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 13000), ylabel="Proteins", xlabel="count")
sns.despine(left=True, bottom=True)
summary = DataFrameSummary(df_raw)
summary.corr
# Compute the correlation matrix
df_noid = df_raw.drop(columns=['Id'])
corr = df_noid.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(160, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
# start small
size, size_validation = 1000, 20
df_keep = df_raw[: size]
df_train = df_keep[: size-size_validation]
df_valid = df_keep[size-size_validation: ]
X_train = loadem(df_train['Id'])
Y_train = df_train[labels]
X_train.shape, Y_train.shape
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D((3, 3), strides=(1, 1), padding='same'))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dense(28))
model.add(Activation('softmax'))

model.build(input_shape=((None, 4, 512, 512)))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])
history = model.fit(X_train, Y_train, batch_size=20, epochs=1, verbose=1)
def showImagesHorizontally(list_of_files):
    fig = plt.figure()
    number_of_files = len(list_of_files)
    for i in range(number_of_files):
        a=fig.add_subplot(1,number_of_files,i+1)
        image = imread(list_of_files[i])
        imshow(image,cmap='Greys_r')
        axis('off')
!ls {PATH}/train/ | head -n 6
imgs = loadem_to_df(['00070df0-bbc3-11e8-b2bc-ac1f6b6435d0'])
imgs.shape

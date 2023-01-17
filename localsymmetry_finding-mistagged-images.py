from scipy import ndimage

from scipy import stats

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline

from skimage import io

import seaborn as sns

import numpy as np

import pandas as pd

import cv2
Labels = pd.read_csv('../input/train_v2.csv')
Labels.head()
Labels.tail()
tagnames = list(set(

    [tag for sublist in Labels['tags'].apply(

        lambda tagstring: tagstring.split()) for tag in sublist]))

print('Tags: ' + '%s' % ', '.join(tagnames))
# Useful function for encoding.

def containsTag(tag, taglist):

    if tag in taglist.split():

        return 1

    else:

        return 0
for tag in tagnames:

    Labels[tag] = Labels['tags'].apply(

        lambda taglist: containsTag(tag, taglist))
Labels.head()
Labels.tail()
def load_image(ImageName):

    """Convert an image given by filename to a numpy array.



    Note, image sizes are expected to be 256 pixels by 256 pixels

    """

    try:

        image_data = io.imread(ImageName).astype(float)

        if image_data.shape != (256, 256, 4):

            raise Exception('Unexpected image shape: %s' %

                            str(image_data.shape))

    except IOError as e:

        print('Could not read:', image, ':', e, '.')

    return image_data
Labels['jpgFilename'] = Labels['image_name'].apply(

    lambda name: '../input/train-jpg/' + name + '.jpg')

Labels['tifFilename'] = Labels['image_name'].apply(

    lambda name: '../input/train-tif-v2/' + name + '.tif')
# Check to see if there is any duplicate or missing atmospheric conditions.

Labels[Labels[['clear', 'haze', 'partly_cloudy', 'cloudy']].sum(axis=1) != 1]
imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_24448.jpg')

axis.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

imageplot.suptitle('Image Number: 24448.')
# A comparison water image with haze to see if there is haze.

imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_3561.jpg')

axis.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

imageplot.suptitle('Image Number: 3561 - water under haze conditions.')
# A comparison water image without haze.

imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_2750.jpg')

axis.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

imageplot.suptitle('Image Number: 2750 - water under clear conditions.')
Labels.loc[24448, 'clear'] = 1

Labels.loc[24448, 'tags'] = 'clear water'
# I will use an isin trick to not have to list all of the encoded tags.

Labels[(Labels[Labels.columns[

    Labels.columns.isin(tagnames)]].sum(axis=1) > 1) & (Labels['cloudy'] == 1)]
Labels[(Labels[Labels.columns[

    Labels.columns.isin(tagnames)]].sum(axis=1) == 1) & (Labels['cloudy'] == 0)]
imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_21276.jpg')

axis.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

imageplot.suptitle('Image Number: 21276.')
Labels.loc[21276, 'primary'] = 1

Labels.loc[21276, 'tags'] = 'partly_cloudy primary'
ConditionCounts = Labels[['clear', 'haze', 'partly_cloudy', 'cloudy']].sum().sort_values(

    ascending=False)

ConditionCounts = ConditionCounts.reset_index(name='Count')

fig = sns.barplot(x='index', y='Count', data=ConditionCounts)

fig.set(xlabel='Condition Type', ylabel='Count', title='Images by Condition')
typeTags = ['artisinal_mine', 'blow_down', 'water', 'cultivation', 'road', 'agriculture',

            'bare_ground', 'blooming', 'selective_logging', 'habitation', 'conventional_mine',

            'slash_burn', 'primary']
TypeCounts = Labels[Labels.columns[Labels.columns.isin(typeTags)]].sum().sort_values(

    ascending=False)

TypeCounts = TypeCounts.reset_index(name='Count')

fig = plt.figure(figsize=(12, 4))

fig = sns.barplot(x='index', y='Count', data=TypeCounts)

fig.set(xlabel='Type', ylabel='Count', title='Images by Type')

# Rotate the tick-labels

for ticklabel in fig.get_xticklabels():

    ticklabel.set_rotation(45)
# Focusing on the rare types:

RareTypeCounts = Labels[['artisinal_mine', 'blow_down', 'bare_ground', 'blooming',

                         'selective_logging', 'conventional_mine',

                         'slash_burn']].sum().sort_values(ascending=False)

RareTypeCounts = RareTypeCounts.reset_index(name='Count')

fig = plt.figure(figsize=(8, 4))

fig = sns.barplot(x='index', y='Count', data=RareTypeCounts)

fig.set(xlabel='Type', ylabel='Count', title='Images by Less Common Types')

# Rotate the tick-labels

for ticklabel in fig.get_xticklabels():

    ticklabel.set_rotation(45)
for condition in ['clear', 'haze', 'partly_cloudy']:

    _ConditionalTypeCounts = Labels[Labels.columns[

        Labels.columns.isin(typeTags)]][Labels[condition] == 1].sum().sort_values(

        ascending=False)

    _ConditionalTypeCounts = _ConditionalTypeCounts.reset_index(name='Count')

    fig = plt.figure(figsize=(12, 4))

    fig = sns.barplot(x='index', y='Count', data=_ConditionalTypeCounts)

    fig.set(xlabel='Type', ylabel='Count',

            title='Images by Type for Condition %s' % condition)

    # Rotate the tick-labels

    for ticklabel in fig.get_xticklabels():

        ticklabel.set_rotation(45)
def plot_image(tifImage, jpgImage, tag, imgNum):

    imageplot = plt.figure(figsize=(10, 6))

    # Plot Red Band

    axis1 = imageplot.add_axes([0, .5, .2, .4])

    axis1.imshow(tifImage[:, :, 2], cmap='Reds')

    axis1.set_title('Red')

    # Plot Green Band

    axis2 = imageplot.add_axes([.25, .5, .2, .4])

    axis2.imshow(tifImage[:, :, 1], cmap='Greens')

    axis2.set_title('Green')

    # Plot Blue Band

    axis3 = imageplot.add_axes([.5, .5, .2, .4])

    axis3.imshow(tifImage[:, :, 0], cmap='Blues')

    axis3.set_title('Blue')

    # Plot NIR Band

    axis4 = imageplot.add_axes([.75, .5, .2, .4])

    axis4.imshow(tifImage[:, :, 3], cmap='magma')

    axis4.set_title('NIR')

    # Plot image

    axis5 = imageplot.add_axes([0, 0, .2, .4])

    axis5.imshow(cv2.cvtColor(jpgImage, cv2.COLOR_BGR2RGB))

    axis5.set_title('JPG Image')

    # Plot color histogram

    axis6 = imageplot.add_axes([.3, .05, .5, .3])

    axis6.hist([tifImage[:, :, 2], tifImage[:, :, 1], tifImage[:, :, 0], tifImage[:, :, 3]],

               bins=100, label=['r', 'g', 'b', 'nir'],

               color=['red', 'green', 'blue', 'magenta'], histtype='step')

    axis6.legend()

    axis6.set_title('Color Histogram')

    imageplot.suptitle('Image Number: %s. Tags: %s.' % (imgNum, tag))
image_sample = Labels.sample(6)

for img in image_sample.index:

    tag = image_sample.loc[img]['tags']

    tifImage = load_image(image_sample.loc[img]['tifFilename'])

    jpgImage = cv2.imread(image_sample.loc[img]['jpgFilename'])

    plot_image(tifImage, jpgImage, tag, img)
def get_band_information(image):

    """Given a single color band of an image, returns the following tuple:

    (band_mean, band_median, band_std, band_max, band_min, band_kurtosis, band_skewness)

    """

    band = image[:, :].ravel()

    return (np.mean(band), np.median(band), np.std(band), np.max(band), np.min(band),

            stats.kurtosis(band), stats.skew(band))
# We will use ColorStats as an array to hold our color information and later join it to the data frame.

n, _ = Labels.shape

ColorStats = np.zeros((n, 28))

for ind in Labels.index:

    imageName = Labels['tifFilename'].loc[ind]

    current_image = load_image(imageName)

    r_stats = get_band_information(current_image[:, :, 2])

    g_stats = get_band_information(current_image[:, :, 1])

    b_stats = get_band_information(current_image[:, :, 0])

    n_stats = get_band_information(current_image[:, :, 3])

    ColorStats[ind, :7] = list(r_stats)  # columns 0-6 are for red

    ColorStats[ind, 7:14] = list(g_stats)  # columns 7-13 are for green

    ColorStats[ind, 14:21] = list(b_stats)  # columns 14-20 are for blue

    ColorStats[ind, 21:] = list(n_stats)  # columns 21-27 are for near-ir

    if ind % 5000 == 0:

        print('Processed %s images.' % ind)
names = []

for color in ['r', 'g', 'b', 'n']:

    for stat in ['mean', 'median', 'std', 'max', 'min', 'kurtosis', 'skewness']:

        names.append(color + '_' + stat)
ColorStatsDF = pd.DataFrame(ColorStats, columns=names)
ColorStatsDF.head()
Labels = pd.concat([Labels, ColorStatsDF], axis=1)
Labels.head()
sns.distplot(Labels[Labels['cloudy'] == 1]['b_std'], kde=False)
sns.boxplot(Labels[Labels['cloudy']==1]['b_std'])
Labels[(Labels['cloudy'] == 1) & (Labels['b_std'] > 8000)]
plot_image(

    io.imread(Labels['tifFilename'].loc[352]), 

    cv2.imread(Labels['jpgFilename'].loc[352]), 

    Labels['tags'].loc[352], 352)
cv2imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_352.jpg')

axis.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

imageplot.suptitle('Image Number: 352.')
plot_image(io.imread(Labels['tifFilename'].loc[22748]), 

           cv2.imread(Labels['jpgFilename'].loc[22748]), 

           Labels['tags'].loc[22748], 22748)
imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_22748.jpg')

axis.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

imageplot.suptitle('Image Number: 22748.')
plot_image(io.imread(Labels['tifFilename'].loc[2550]), 

           cv2.imread(Labels['jpgFilename'].loc[2550]), 

           Labels['tags'].loc[2550], 2550)
imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_2550.jpg')

axis.imshow(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

imageplot.suptitle('Image Number: 2550.')
# Fix the miss labelling

Labels.loc[2550, 'cloudy'] = 0

Labels.loc[2550, 'partly_cloudy'] = 1

Labels.loc[2550, 'primary'] = 1

Labels.loc[2550, 'tags'] = 'partly_cloudy primary'
sns.distplot(Labels[Labels['partly_cloudy'] == 1]['n_min'], kde=False)
sns.boxplot(Labels[Labels['partly_cloudy'] == 1]['n_min'])
Labels[(Labels['partly_cloudy'] == 1) & (Labels['n_min'] > 14000)]
plot_image(io.imread(Labels['tifFilename'].loc[35252]), 

           cv2.imread(Labels['jpgFilename'].loc[35252]), 

           Labels['tags'].loc[35252], 35252)
imageplot = plt.figure(figsize=(6, 6))

axis = imageplot.add_axes([0, 0, .9, .9])

_image = cv2.imread('../input/train-jpg/train_35252.jpg')

axis.imshow(_image[:, :, 0], cmap = 'Reds')

imageplot.suptitle('Image Number: 35252.')
# Make sure this is not an issue in the data frame

Labels[['tifFilename', 'jpgFilename']].loc[35252]
# Create a list of statistics that we can log transform. We will not log transform skew or kurtosis.

lognames = []

for color in ['r', 'g', 'b', 'n']:

    for stat in ['mean', 'median', 'std', 'max', 'min']:

        lognames.append(color + '_' + stat)





# Helper functions for investigating band statistics.

def plot_band_stats(DataFrame, name='Unknown'):

    """Given a DataFrame as a data frame of Labels, will plot all band statistics for that data frame.



    Parameters

        DataFrame: The data frame of Labels to plot

        name: Name for the statistics.



    Example

    plot_band_stats(Labels['primary'], 'primary') will plot all band statistics for images

    that are tagged 'primary'

    """

    for stat in names:

        fig = plt.figure(figsize=(9, 4))

        histogram = plt.subplot(2, 1, 1)

        sns.distplot(DataFrame[stat], kde=False, bins=100)

        boxplot = plt.subplot(2, 1, 2)

        sns.boxplot(DataFrame[stat])

        plt.suptitle(stat + ' for: ' + name)





def plot_logband_stats(DataFrame, name='Unknown'):

    """Given a DataFrame as a data frame of Labels, will plot logged band statistics for that data frame.

    Note: skewness and kurotosis are not plotted.



    Parameters

        DataFrame: The data frame of Labels to plot

        name: Name for the statistics.



    Example

    plot_band_stats(Labels['primary'], 'primary') will plot all logged band statistics for images

    that are tagged 'primary'

    """

    for stat in lognames:

        fig = plt.figure(figsize=(9, 4))

        histogram = plt.subplot(2, 1, 1)

        sns.distplot(np.log(1 + DataFrame[stat]), kde=False, bins=100)

        boxplot = plt.subplot(2, 1, 2)

        sns.boxplot(np.log(1 + DataFrame[stat]))

        plt.suptitle('logged ' + stat + ' for: ' + name)
ClearPrimary = Labels[Labels['tags'] == 'clear primary']

plot_band_stats(ClearPrimary, name='clear primary')
# You can also investigate logged statistics as follows, but the kernel is running out of memory.

# plot_logband_stats(ClearPrimary, name='clear primary')
IQR = stats.iqr(ClearPrimary['n_max'])

Lower = np.percentile(ClearPrimary['n_max'], 25) - 3*IQR

Upper = np.percentile(ClearPrimary['n_max'], 75) + 3*IQR
Candidates = ClearPrimary[(ClearPrimary['n_max'] < Lower) | (ClearPrimary['n_max'] > Upper)]
Candidates.shape # 70 images found.
for img in [221, 1270, 1652, 3518, 4639, 7112, 12645, 14687, 23638, 28381, 34729, 4978, 25513]:

    tag = Candidates.loc[img]['tags']

    tifImage = load_image(Candidates.loc[img]['tifFilename'])

    jpgImage = cv2.imread(Candidates.loc[img]['jpgFilename'])

    plot_image(tifImage, jpgImage, tag, img)
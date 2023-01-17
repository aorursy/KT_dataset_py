# Import common libraries

import os

import glob

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

import openslide

from openslide import OpenSlideError

from IPython.display import Image

import seaborn as sns

import multiprocessing

import datetime



class Time:

  """

  Class for displaying elapsed time.

  """



  def __init__(self):

    self.start = datetime.datetime.now()



  def elapsed_display(self):

    time_elapsed = self.elapsed()

    print("Time elapsed: " + str(time_elapsed))



  def elapsed(self):

    self.end = datetime.datetime.now()

    time_elapsed = self.end - self.start

    return time_elapsed



# Open a slide

def open_slide(filename):

    """

    Open a whole-slide image (*.svs, etc).

    :filename : Name of the slide file.

    return: an OpenSlide object representing a whole-slide image.

    """

    try:

        slide = openslide.open_slide(filename)

    except OpenSlideError:

        slide = None

    except FileNotFoundError:

        slide = None

    return slide    
# PARAMETERS

BASE_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'

OUTPUT_DIR = './'

TRAIN_DIR = os.path.join(BASE_DIR, "train_images")

TRAIN_EXT = ".tiff"

MASK_DIR = os.path.join(BASE_DIR, "train_label_masks")

MASK_EXT = "_mask.tiff"
# Number slide > Mask so we take the mask as the minimum value

train = glob.glob1(TRAIN_DIR, "*" + TRAIN_EXT)

label = glob.glob1(MASK_DIR, "*" + MASK_EXT)



# Keep only image_id

train = [x[:-len(TRAIN_EXT)] for x in train]

label = [y[:-len(MASK_EXT)] for y in label]



len(train), len(label)
# Add filenames to dataframe

train_df = pd.read_csv(BASE_DIR + 'train.csv')



# Add train file column for each existing file in train folder

train_df['train_file'] = list(map(lambda x : x + TRAIN_EXT if x in set(train) else '', 

                              train_df['image_id']))

# Add label file column for each existing file in mask folder

train_df['label_file'] = list(map(lambda y : y + MASK_EXT if y in set(label) else '', 

                              train_df['image_id']))

train_df.head()
# Split dataframe by provider / we keep radboud scoring because their mask labels are more details

print('Dataframe original:', len(train_df))

train_radboud = train_df[train_df['data_provider'] == 'radboud'].copy()

print('Dataframe after split:', len(train_radboud))



# Keep only row with both train and label file

train_radboud = train_radboud[train_radboud['train_file'] != '']

print('Dataframe after file select:', len(train_radboud))

train_radboud = train_radboud[train_radboud['label_file'] != '']

print('Dataframe after label select:', len(train_radboud))



train_radboud.head()
# Check time to open and close a slide

file = train_radboud['train_file'].values[0]

filepath = os.path.join(TRAIN_DIR, file)



# Open

t = Time()

print('Open slide')

biopsy = open_slide(filepath)

t.elapsed_display()



# Close

t = Time()

print('Close slide')

biopsy.close()

t.elapsed_display()



# OBSERVATION : <20ms to open and close a slide
# Check on all the dataset

t = Time()

files = train_radboud['train_file'].values[:1]

for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    # Do something

    biopsy.close()

print('Open and close', len(files),'slides')

t.elapsed_display()



# OBSERVATION : ~1-2 min to open and close more 5000 slides => ~20ms by slide
# Keep only ISUP grade = 5

train_radboud5 = train_radboud[train_radboud['isup_grade'] == 5].copy()

print('Dataframe after grade select:', len(train_radboud5))

train_radboud5.head()
# Check times on this dataset

t = Time()

files = train_radboud5['train_file'].values

for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    # Do something

    biopsy.close()

print('Open and close', len(files),'slides')

t.elapsed_display()



# OBSERVATION : ~10s to open and close 964 slides
# Check time to open and close the lowest level image

t = Time()

files = train_radboud5['train_file'].values[:1]

for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]

    sample = biopsy.read_region((0, 0), level, dimensions)

    # Close

    biopsy.close()

    sample = None

print('Open, read image and close', len(files),'slides')

t.elapsed_display()



# OBSERVATION 2: <1s to open, read and close 964 slides

# OBSERVATION 2: ~40s to open, read and close 964 slides
# Check time to open, save and close the lowest level image

DEST_TRAIN_DIR = 'train_png'

DEST_TRAIN_EXT = '.png'



t = Time()

files = train_radboud5['train_file'].values[:1]

for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]

    sample = biopsy.read_region((0, 0), level, dimensions)

    # Save

    if not os.path.exists(DEST_TRAIN_DIR):

        os.makedirs(DEST_TRAIN_DIR)

    sample.save(os.path.join(DEST_TRAIN_DIR, file + DEST_TRAIN_EXT))

    # Close

    biopsy.close()

    sample = None

print('Open, read, save and close', len(files),'slides')

t.elapsed_display()



# OBSERVATION 1: ~400ms to open, read, save and close 1 slide

# OBSERVATION 2: < 3min to open, read, save and close 964 slides
# Count the proportion of blank white pixel (= 255) by slide

t = Time()

files = train_radboud5['train_file'].values

white_pixel = []



for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]

    sample = biopsy.read_region((0, 0), level, dimensions)

    num_pixels = dimensions[0]*dimensions[1]

    sample = sample.convert("1") #Convert to black and white

    white_pixel.append(np.count_nonzero(sample)/num_pixels)  

    # Close

    biopsy.close()

    sample = None

print('Open, read, save and close', len(files),'slides')

t.elapsed_display()



train_radboud5['white_proportion'] = white_pixel

white_pixel = None



# OBSERVATION 1: varying the level definition as no impact on the white proportion so we can keep the lowest level

# OBSERVATION 2: 50s to count white pixel on 964 slides
train_radboud5.describe()
# Displaying a very 'blank' slide

t = Time()

files = train_radboud5.loc[train_radboud5['white_proportion']>0.98]['train_file'].values[:1]



for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]

    sample = biopsy.read_region((0, 0), level, dimensions)

    display(sample)

    # Close

    biopsy.close()

    sample = None

print('Open, show, save and close', len(files),'slides')

t.elapsed_display()



# OBSERVATION : 250ms to display a slide as low definition
# Draw a grid on an image

# https://stackoverflow.com/questions/20368413/draw-grid-lines-over-an-image-in-matplotlib

import matplotlib.ticker as plticker



# Open image file

t = Time()

files = train_radboud5['train_file'].values[:1]



for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]

    sample = biopsy.read_region((0, 0), level, dimensions)



    # Resolution

    dpi=100.



    # Set up figure

    fig=plt.figure(figsize=(float(sample.size[0])/dpi,float(sample.size[1])/dpi),dpi=dpi)

    ax=fig.add_subplot(111)



    # Set the gridding interval: here we use the major tick interval

    interval=32

    loc = plticker.MultipleLocator(base=interval)

    ax.xaxis.set_major_locator(loc)

    ax.yaxis.set_major_locator(loc)

    ax.xaxis.set_ticklabels([])

    ax.yaxis.set_ticklabels([])



    # Add the grid

    ax.grid(which='major', axis='both', linestyle='-')



    # Add the image

    ax.imshow(sample)



    # Find number of gridsquares in x and y direction

    nx=abs(int(float(ax.get_xlim()[1]-ax.get_xlim()[0])/float(interval)))

    ny=abs(int(float(ax.get_ylim()[1]-ax.get_ylim()[0])/float(interval)))



    # Add some labels to the gridsquares

    for i in range(nx):

        x=interval/2.+float(i)*interval

        ax.text(x,interval/2,i,color='black',ha='center',va='center')

    for j in range(ny):

        y=interval/2+j*interval

        ax.text(interval/2,y,j,color='black',ha='center',va='center')





    # Save the figure

    #fig.savefig('myImageGrid.tiff',dpi=my_dpi)

    

    # Close

    biopsy.close()

    sample = None

print('Open, draw grid and close', len(files),'slides')

t.elapsed_display()



# OBSERVATION : 210ms to draw a grid on slide at low definition
# See in detail the (19,5) tile on different level size

t = Time()

files = train_radboud5['train_file'].values[:1]

index = (12, 47)

interval = 32



for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    for level in range(biopsy.level_count):

        scale = int(16/biopsy.level_downsamples[level]) # Scale factor of the given level

        size = interval*scale # Tile size depend on scale factor

        dimensions = (size, size)

        x, y = index[0]*interval*16, index[1]*interval*16 #Localisation from the level 0 => * max scale interval to get coordinate

        sample = biopsy.read_region((x, y), level, dimensions)



        # Display

        print('tile:', index, '- level:', level, '- scale:', scale,'- size:', size)

        display(sample)



    # Close

    biopsy.close()

    sample = None

print('Open, show tiles and close', len(files),'slides')

t.elapsed_display()



# OBSERVATION : 330ms to display tiles on each level definition
# Display a heatmap of tile color intensities

t = Time()

files = train_radboud5['train_file'].values[:1]

white_pixel = []

interval = 32



for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)    

    # Open

    biopsy = open_slide(filepath)

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]



    # Get number of gridsquares in x and y direction

    nx=int(dimensions[0]/interval)

    ny=int(dimensions[1]/interval)

    tiles = np.zeros((nx, ny))



    # Browse each tiles

    level = 1

    scale = 4

    size = interval*scale # Tile size depend on scale factor

    dimensions = (size, size)

    num_pixels = dimensions[0]*dimensions[1]

    

    for i in range(nx):

        for j in range(ny):  

            x, y = i*interval*16, j*interval*16 #Localisation from the level 0 => * max scale interval to get coordinate

            sample = biopsy.read_region((x, y), level, dimensions)

            sample = sample.convert("1") #Convert to black and white

            tiles[i][j] = 1-np.count_nonzero(sample)/num_pixels

    white_pixel.append(tiles)

    

    # Close

    biopsy.close()

    sample = None

    

print('Open, score tiles and close', len(files),'slides')



# Generate a heatmap

grid = white_pixel[0]

sns.set(style="white")

plt.subplots(figsize=(grid.shape[0]/5, grid.shape[1]/5))



mask = np.zeros_like(grid)

mask[np.where(grid < 0.1)] = True #Mask blank tiles



sns.heatmap(grid.T, square=True, linewidths=.5, mask=mask.T, cbar=False, vmin=0, vmax=1, cmap="Reds")

plt.show()

print('Not-blank tiles:', np.count_nonzero(grid), 'on', grid.size, 'total tiles')

grid = None

white_pixel = None



t.elapsed_display()

# OBSERVATION: 1.5s to count white pixel on 1 slide
# Generate tiles for one slide

t = Time()

files = train_radboud5['train_file'].values[:1]

tiles = []

interval = 32



for file in tqdm(files):

    filepath = os.path.join(TRAIN_DIR, file)

    

    # Open

    biopsy = open_slide(filepath)

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]



    # Get number of gridsquares in x and y direction

    nx=int(dimensions[0]/interval)

    ny=int(dimensions[1]/interval)

    #tiles = np.zeros((nx, ny))



    # Browse each tiles

    level = 1

    scale = 4

    size = interval*scale # Tile size depend on scale factor

    dimensions = (size, size)

    num_pixels = dimensions[0]*dimensions[1]

    

    for i in range(nx):

        for j in range(ny):  

            x, y = i*interval*16, j*interval*16 #Localisation from the level 0 => * max scale interval to get coordinate

            sample = biopsy.read_region((x, y), level, dimensions)

            sample = sample.convert("1") #Convert to black and white

            score = 1-np.count_nonzero(sample)/num_pixels

            # Keep only not blank tiles

            if score > 0.1:

                tiles.append({'image_id':file[:-len('.tiff')], 'tile':(i,j), 'x':x, 'y':y, 'level':level, 'size':size})

    

    # Close

    biopsy.close()

    sample = None

    

print('Extract',len(tiles),'tiles from', len(files),'slides')

t.elapsed_display()



# OBSERVATION 1: ~1,5s to extract and score tiles on 1 slides

# OBSERVATION 2: to keep an equivalent time, we could use multiprocessing on a larger dataset of 5000 slides.
# Generate tiles from files in a given range

def generate_tiles(file):

    interval = 32

    tiles = []

    filepath = os.path.join(TRAIN_DIR, file)



    # Open

    biopsy = open_slide(filepath)

    

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]



    # Get number of gridsquares in x and y direction

    nx=int(dimensions[0]/interval)

    ny=int(dimensions[1]/interval)

    #tiles = np.zeros((nx, ny))



    # Browse each tiles

    level = 1

    scale = 4

    size = interval*scale # Tile size depend on scale factor

    dimensions = (size, size)

    num_pixels = dimensions[0]*dimensions[1]



    for i in range(nx):

        for j in range(ny):  

            x, y = i*interval*16, j*interval*16 #Localisation from the level 0 => * max scale interval to get coordinate

            sample = biopsy.read_region((x, y), level, dimensions)

            sample = sample.convert("1") #Convert to black and white

            score = 1-np.count_nonzero(sample)/num_pixels

            # Keep only not blank tiles

            if score > 0.1:

                tiles.append({'image_id':file[:-len('.tiff')], 'tile':(i,j), 'x':x, 'y':y, 'level':level, 'size':size})



    # Close

    biopsy.close()

    sample = None

    return tiles
# Extract tiles for all slides (with multiprocessing)

t = Time()

files = train_radboud5['train_file'].values



# Processes available

num_processes = multiprocessing.cpu_count()

pool = multiprocessing.Pool(num_processes)



# Image per process split

num_files = len(files)

if num_processes > num_files:

    num_processes = num_files

files_per_process = num_files / num_processes



print("Number of processes: " + str(num_processes))

print("Number of files: " + str(num_files))



# start tasks pooling

tiles = []



def get_tiles(result):

    tiles.append(result)

    

for file in tqdm(files):

    result = pool.apply_async(generate_tiles, args = (file,), callback = get_tiles)



pool.close()

pool.join()



tiles = np.concatenate(tiles)

print('Extract',len(tiles),'tiles from', len(files),'slides')

t.elapsed_display()



# OBSERVATION 1: ~1,5s to extract and score tiles from 1 slides

# OBSERVATION 2: ~10m to extract and score tiles of 961 slides from 1 process

# OBSERVATION 3: ~5m to extract and score tiles of 961 slides from 4 process
# Export tiles for further usage

t = Time()

tiles_df = pd.DataFrame(tiles.tolist())

tiles_df.to_csv('PANDA_tiles_EDA.csv')

t.elapsed_display()
# Import tiles dataframe for checkpoint purpose

tiles_df = None # Free memory

# tiles_df = pd.read_csv('/kaggle/input/panda-tiles-EDA/PANDA_tiles_EDA.csv', index_col=0)

# tiles_df.head()
# Display a mask with tiles scoring

t = Time()

files = train_radboud5['label_file'].values[:1]

tiles = []

interval = 32



for file in tqdm(files):

    filepath = os.path.join(MASK_DIR, file)    

    # Open

    gleason = open_slide(filepath)

    

    # Read lowest definition image

    level = gleason.level_count - 1

    dimensions = gleason.level_dimensions[level]



    # Get number of tiles in x and y direction

    nx=int(dimensions[0]/interval)

    ny=int(dimensions[1]/interval)

    labels = np.zeros((nx, ny, 6)) #tiles with score dimension



    # Browse each tiles

    level = 1

    scale = 4

    size = interval*scale # Tile size depend on scale factor

    dimensions = (size, size)

    num_pixels = dimensions[0]*dimensions[1]

    

    for i in range(nx):

        for j in range(ny):  

            x, y = i*interval*16, j*interval*16 #Localization from the level 0 => * max scale interval to get coordinate

            sample = gleason.read_region((x, y), level, dimensions)

            sample = np.array(sample.convert('RGB'))

            labels[i][j] = np.zeros(6, dtype='uint') #Create an empty score list

            key, value = np.unique(sample[:,:,0], return_counts=True) # Count by pixel score present on the first color channel

            scores = dict(zip(key, value)) #Create a score dict

            for k in scores.keys():

                labels[i][j][k] = scores[k]/num_pixels #Update score list

    tiles.append(labels)

    

    # Close

    gleason.close()

    #sample = None

    

print('Open, score tiles and close', len(files),'slides')



# Generate a heatmap

grade = 5

grid = tiles[0][:,:,grade] # Get negative zone score

sns.set(style="white")

plt.subplots(figsize=(grid.shape[0]/5, grid.shape[1]/5))



mask = np.zeros_like(grid)

mask[np.where(grid < 0.1)] = True #Mask threshold



sns.heatmap(grid.T, square=True, linewidths=.5, mask=mask.T, cbar=False, vmin=0, vmax=1, cmap="Reds")

plt.show()



print('Tiles with Gleason grade', grade, '> 10% on', grid.size, 'tiles')

grid = None



t.elapsed_display()

# OBSERVATION: 2s to score tiles from 1 slide
# Generate score tiles from files and masks

def generate_tiles_labels(file):

    interval = 32

    tiles = []

    filepath = os.path.join(TRAIN_DIR, file)

    image_id = file[:-len(TRAIN_EXT)]

    maskpath = os.path.join(MASK_DIR, image_id + MASK_EXT)



    # Open files

    biopsy = open_slide(filepath)

    mask = open_slide(maskpath)

    

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]



    # Get number of gridsquares in x and y direction

    nx=int(dimensions[0]/interval)

    ny=int(dimensions[1]/interval)

    #tiles = np.zeros((nx, ny))



    # Browse each tiles

    level = 1

    scale = 4

    size = interval*scale # Tile size depend on scale factor

    dimensions = (size, size)

    num_pixels = dimensions[0]*dimensions[1]



    for i in range(nx):

        for j in range(ny):  

            x, y = i*interval*16, j*interval*16 #Localization from the level 0 => * max scale interval to get coordinate

            

            # Read biopsy file

            sample = biopsy.read_region((x, y), level, dimensions)

            sample = sample.convert("1") #Convert to black and white

            score = 1-np.count_nonzero(sample)/num_pixels



            # Keep only not blank tiles

            if score > 0.1:

                # Read mask file

                sample = mask.read_region((x, y), level, dimensions)

                sample = np.array(sample.convert('RGB'))

                

                key, value = np.unique(sample[:,:,0], return_counts=True) # Count by pixel score present on the first color channel

                scores = dict(zip(key, value)) #Create a score dict

                

                PREFIX = 'gleason_'

                labels = {PREFIX+str(k) : 0 for k in range(6)} #Create an empty score list from 0 to 5

                for k in scores.keys():

                    labels[PREFIX+str(k)] = scores[k]/num_pixels #Update score list

                

                # Add tile

                tile = {'image_id':file[:-len('.tiff')], 'tile':(i,j), 'x':x, 'y':y, 'level':level, 'size':size,}   

                tiles.append({**tile, **labels})



    # Close

    biopsy.close()

    mask.close()

    sample = None

    return tiles
# Testing generate_tiles_labels

file = train_radboud5['train_file'].values[0]

tiles = generate_tiles_labels(file)

test_df = pd.DataFrame(tiles)

test_df.describe()
test_df = None # Free memory
# Extract label tiles for all slides (with multiprocessing)

t = Time()

files = train_radboud5['train_file'].values



# Processes available

num_processes = multiprocessing.cpu_count()

pool = multiprocessing.Pool(num_processes)



# Image per process split

num_files = len(files)

if num_processes > num_files:

    num_processes = num_files

files_per_process = num_files / num_processes



print("Number of processes: " + str(num_processes))

print("Number of files: " + str(num_files))



# start tasks pooling

tiles = []



def get_tiles(result):

    tiles.append(result)

    

for file in tqdm(files):

    result = pool.apply_async(generate_tiles_labels, args = (file,), callback = get_tiles)



pool.close()

pool.join()



tiles = np.concatenate(tiles)

print('Extract',len(tiles),'tiles from', len(files),'slides')

t.elapsed_display()



# OBSERVATION 1: ~2s to extract and score tiles from 1 slides

# OBSERVATION 2: ~6m to extract and score tiles from 964 slides on 4 process
# Use this to interrupt the multiprocess pool and free your CPUs after a 'Cancel run' command

pool.terminate()

pool.close()

pool.join()
# Export tiles for further usage

t = Time()

tiles_labels_df = pd.DataFrame(tiles.tolist())

tiles_labels_df.to_csv('PANDA_tiles_labels_EDA.csv')

print('Export dataframe')

t.elapsed_display()
# Import tiles with labels dataframe for checkpoint purpose

tiles_labels_df = None # Free memory

# tiles_labels_df = pd.read_csv('/kaggle/input/panda-tiles-labels_EDA/PANDA_tiles_labels_EDA.csv', index_col=0)

# tiles_labels_df.head()
# DEPENDANCIES ###########################################################################

import os

import glob

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

from tqdm.notebook import tqdm

import openslide

from openslide import OpenSlideError

from IPython.display import Image

import seaborn as sns

import multiprocessing

import datetime

# / DEPENDANCIES #########################################################################

print("Dependencies loaded")



# UTILITIES ##############################################################################

class Time:

  """

  Class for displaying elapsed time.

  """



  def __init__(self):

    self.start = datetime.datetime.now()



  def elapsed_display(self):

    time_elapsed = self.elapsed()

    print("Time elapsed: " + str(time_elapsed))



  def elapsed(self):

    self.end = datetime.datetime.now()

    time_elapsed = self.end - self.start

    return time_elapsed

# / UTILITIES ############################################################################

print('Utilities loaded')



# PARAMETERS #############################################################################

BASE_DIR = '/kaggle/input/prostate-cancer-grade-assessment/'

OUTPUT_DIR = './'

TRAIN_DIR = os.path.join(BASE_DIR, "train_images")

TRAIN_EXT = ".tiff"

MASK_DIR = os.path.join(BASE_DIR, "train_label_masks")

MASK_EXT = "_mask.tiff"



print("Parameters loaded")

# /PARAMETERS ############################################################################



# DATASET ################################################################################

# Get train/label slides ID

train = glob.glob1(TRAIN_DIR, "*" + TRAIN_EXT)

label = glob.glob1(MASK_DIR, "*" + MASK_EXT)



# Keep only image_id

train = [x[:-len(TRAIN_EXT)] for x in train]

label = [y[:-len(MASK_EXT)] for y in label]



# Add filenames to dataframe

train_df = pd.read_csv(BASE_DIR + 'train.csv')



# Add train file column for each existing file in train folder

train_df['train_file'] = list(map(lambda x : x + TRAIN_EXT if x in set(train) else '', 

                              train_df['image_id']))

# Add label file column for each existing file in mask folder

train_df['label_file'] = list(map(lambda y : y + MASK_EXT if y in set(label) else '', 

                              train_df['image_id']))



# Split dataframe by provider / we keep radboud scoring because their mask labels are more details

print('Dataframe original:', len(train_df))

train_radboud = train_df[train_df['data_provider'] == 'radboud'].copy()

print('Dataframe after provider select:', len(train_radboud))

# Keep only row with both train and label file

train_radboud = train_radboud[train_radboud['train_file'] != '']

print('Dataframe after file select:', len(train_radboud))

train_radboud = train_radboud[train_radboud['label_file'] != '']

print('Dataframe after label select:', len(train_radboud))



# Release memory

train_df = None

# / DATASET ##############################################################################





# FUNCTIONS ##############################################################################

# Open a slide

def open_slide(filename):

    """

    Open a whole-slide image (*.svs, etc).

    :filename : Name of the slide file.

    return: an OpenSlide object representing a whole-slide image.

    """

    try:

        slide = openslide.open_slide(filename)

    except OpenSlideError:

        slide = None

    except FileNotFoundError:

        slide = None

    return slide    



# Generate score tiles from files and masks

def generate_tiles_labels(file):

    """

    Generate a list of tiles with coordonnate and label from file/mask whole-slide image .tiff

    :file : Name of the slide file (must start and end with define directory and extension)

    return: a list of dictionnary tiles with gleason labels

    """

    interval = 32

    tiles = []

    filepath = os.path.join(TRAIN_DIR, file)

    image_id = file[:-len(TRAIN_EXT)]

    maskpath = os.path.join(MASK_DIR, image_id + MASK_EXT)



    # Open files

    biopsy = open_slide(filepath)

    mask = open_slide(maskpath)

    

    # Read lowest definition image

    level = biopsy.level_count - 1

    dimensions = biopsy.level_dimensions[level]



    # Get number of gridsquares in x and y direction

    nx=int(dimensions[0]/interval)

    ny=int(dimensions[1]/interval)

    #tiles = np.zeros((nx, ny))



    # Browse each tiles

    level = 1

    scale = 4

    size = interval*scale # Tile size depend on scale factor

    dimensions = (size, size)

    num_pixels = dimensions[0]*dimensions[1]



    for i in range(nx):

        for j in range(ny):  

            x, y = i*interval*16, j*interval*16 #Localization from the level 0 => * max scale interval to get coordinate

            

            # Read biopsy file

            sample = biopsy.read_region((x, y), level, dimensions)

            sample = sample.convert("1") #Convert to black and white

            score = 1-np.count_nonzero(sample)/num_pixels #Normalize the value between 0 and 1 (0=white, 1=black)



            # Keep only not empty tiles

            if score > 0.1:

                # Read mask file

                sample = mask.read_region((x, y), level, dimensions)

                sample = np.array(sample.convert('RGB'))

                

                key, value = np.unique(sample[:,:,0], return_counts=True) # Count by pixel score present on the first color channel

                scores = dict(zip(key, value)) #Create a score dict

                

                PREFIX = 'gleason_'

                labels = {PREFIX+str(k) : 0 for k in range(6)} #Create an empty score list from 0 to 5

                for k in scores.keys():

                    labels[PREFIX+str(k)] = scores[k]/num_pixels #Update score list

                

                # Add tile

                tile = {'image_id':file[:-len('.tiff')], 'tile':(i,j), 'x':x, 'y':y, 'level':level, 'size':size,}   

                tiles.append({**tile, **labels})



    # Close

    biopsy.close()

    mask.close()

    sample = None

    return tiles

print('Functions loaded')

# / FUNCTIONS ############################################################################



t = Time() # Launch timer



# EXTRACTION #############################################################################

# Extract label tiles for all slides (with multiprocessing)

print('Start tiles generation...')

files = train_radboud['train_file'].values



# Processes available

num_processes = multiprocessing.cpu_count()

pool = multiprocessing.Pool(num_processes)



# Image per process split

num_files = len(files)

if num_processes > num_files:

    num_processes = num_files

files_per_process = num_files / num_processes



print("Number of processes: " + str(num_processes))

print("Number of files: " + str(num_files))



# start tasks pooling

tiles = []



def get_tiles(result):

    tiles.append(result)

    

for file in files:

    result = pool.apply_async(generate_tiles_labels, args = (file,), callback = get_tiles)



pool.close()

pool.join()



tiles = np.concatenate(tiles)

print('Extract',len(tiles),'tiles from', len(files),'slides')



# OBSERVATION: ~30m to extract and score 603602 tiles from 5060 slides on 4 process

# / EXTRACTION ###########################################################################



# OUTPUT #################################################################################

# Export tiles for further usage

tiles_final_df = pd.DataFrame(tiles.tolist())

tiles_final_df.to_csv(OUTPUT_DIR + 'PANDA_tiles_labels_final.csv')

print('Tiles exported')



# OBSERVATION: output csv file of size 72,4Mb

# / OUTPUT ###############################################################################



t.elapsed_display() # Print timer
# Use this to interrupt the multiprocess pool and free your CPUs after a 'Cancel run' command

pool.terminate()

pool.close()

pool.join()
tiles_final_df.head()
tiles_final_df.describe()
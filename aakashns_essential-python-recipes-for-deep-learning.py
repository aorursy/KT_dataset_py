# You can use shell commands with "!"

!ls ../input
# Pipe output to do basic analysis

!ls ../input/train/ | wc -l

!ls ../input/train/ | head
!ls ../input/train/00070df0-bbc3-11e8-b2bc-ac1f6b6435d0_*.png
# Better approach: use pathlib

from pathlib import Path



DATA_DIR = Path('../input')

TRAIN_DIR = DATA_DIR/'train'

TEST_DIR = DATA_DIR/'test'
list(set([str(fn).split('/')[-1].split('_')[0] for fn in TEST_DIR.iterdir()]))[:10]
# Use the full power of Python

test_ids = list(set([str(fn).split('/')[-1].split('_')[0]  for fn in TEST_DIR.iterdir()]))

print('Test IDs:', len(test_ids))

test_ids[:10]
# You can even create directories

SUB_DIR = Path('files/submissions')

SUB_DIR.mkdir(parents=True, exist_ok=True)

!ls files
# You could always use shell commands

LABELS_CSV = DATA_DIR/'train.csv'

!head {LABELS_CSV}
# Enter pandas

import pandas as pd



train_df = pd.read_csv(LABELS_CSV, index_col='Id')

train_df.head(10)
# You can look at a random sample

train_df.sample(10)
# Or get basic information about the data

train_df.info()
# Use Python to your advantage

train_df['Target'] = train_df['Target'].str.split(' ').map(lambda x: list(map(int, x)))

train_df.head(10)
label_names = ["Nucleoplasm", "Nuclear membrane", "Nucleoli", "Nucleoli fibrillar center", 

               "Nuclear speckles", "Nuclear bodies", "Endoplasmic reticulum", 

               "Golgi apparatus", "Peroxisomes", "Endosomes","Lysosomes", 

               "Intermediate filaments", "Actin filaments", "Focal adhesion sites", 

               "Microtubules", "Microtubule ends", "Cytokinetic bridge", "Mitotic spindle", 

               "Microtubule organizing center", "Centrosome", "Lipid droplets", 

               "Plasma membrane", "Cell junctions", "Mitochondria", "Aggresome",   

               "Cytosol", "Cytoplasmic bodies", "Rods & rings"]
import numpy as np



def get_label_freqs(targets, label_names, ascending=None):

    n_classes = len(label_names)

    freqs = np.array([0] * n_classes)

    for lst in targets:

        for c in range(n_classes):

            freqs[c] += c in lst

    data = {

        'name': label_names, 

        'frequency': freqs, 

        'percent': (10000 * freqs / len(targets)).astype(int) / 100.,

    }

    cols = ['name', 'frequency', 'percent']

    df = pd.DataFrame(data, columns=cols)

    if ascending is not None:

        df = df.sort_values(by='frequency', ascending=ascending)

    return df
# Create a frequency table

train_freqs = get_label_freqs(train_df.Target, label_names, ascending=False)

train_freqs
# Visualize the frequency table using a chart

train_freqs.plot(x='name', y='frequency', kind='bar', title='Name vs. Frequency');
# Use logarithmic axis for easier interpretation

train_freqs.plot(x='name', y='frequency', kind='bar', logy=True, title='Name vs. log(Frequency)');
train_sample = "ac39847a-bbb1-11e8-b2ba-ac1f6b6435d0_red.png"
from imageio import imread

import matplotlib.pyplot as plt



# Look at one channel/filter

img0 = imread(str(TRAIN_DIR/train_sample))

print(img0.shape)

plt.imshow(img0)

plt.title(train_sample[0]);
# Use a color map for grayscale images

plt.imshow(img0, cmap="Reds");
# For RGB images, it "just works"

!curl https://www.what-dog.net/Images/faces2/scroll001.jpg -o sample.jpg



img = imread('sample.jpg')

plt.imshow(img);
!ls {TRAIN_DIR}/ac39847a-bbb1-11e8-b2ba-ac1f6b6435d0_*.png
CHANNELS = ['green', 'red', 'blue', 'yellow']



# Load images for multiple channels

def load_image(image_id, channels=CHANNELS, img_dir=TRAIN_DIR):

    image = np.zeros(shape=(len(channels),512,512))

    for i, ch in enumerate(channels):

        image[i,:,:] = imread(str(img_dir/f'{image_id}_{ch}.png'))

    return image
# Plot multiple images in a grid

def show_image_filters(image, title, figsize=(16,5)):

    fig, subax = plt.subplots(1, 4, figsize=figsize)

    # Green channel

    subax[0].imshow(image[0], cmap="Greens")

    subax[0].set_title(title)

    # Red channel

    subax[1].imshow(image[1], cmap="Reds")

    subax[1].set_title("Microtubules")

    # Blue channel

    subax[2].imshow(image[2], cmap="Blues")

    subax[2].set_title("Nucleus")

    # Orange channel

    subax[3].imshow(image[3], cmap="Oranges")

    subax[3].set_title("Endoplasmatic reticulum")

    return subax
# Use the traning data to show appropriate labels

def get_labels(image_id):

    labels = [label_names[x] for x in train_df.loc[image_id]['Target']]

    return ', '.join(labels)
# Look at a sample grid

img_id = 'ac39847a-bbb1-11e8-b2ba-ac1f6b6435d0'

img, title = load_image(img_id), get_labels(img_id)

show_image_filters(img, title);

print(img.shape)
# Combine with pandas to view a random sample

for img_id in train_df.sample(3).index:

    print(img_id)

    img, title = load_image(img_id), get_labels(img_id)

    show_image_filters(img, title)
# Let's define a sophisticated and highly accurate model

def model(inputs):

    return np.random.randn(len(inputs), len(label_names))
# Generate some predictions (logits)

preds = model(test_ids)

print(preds.shape)

print(preds)
# Convert them into probabilities

def sigmoid(x):

    return np.reciprocal(np.exp(-x) + 1) 



probs = sigmoid(preds)

probs
# Convert probabilities into labels

def make_labels(y, thres=0.5):

    return ' '.join([str(i) for i, p in enumerate(y) if p > thres])



make_labels(probs[0])
# Create a pandas dataframe

labels = list(map(make_labels, probs))

sub_df = pd.DataFrame({ 'Id': test_ids, 'Predicted': labels}, columns=['Id', 'Predicted'])

sub_df.head(10)
# Export it to a file and make sure it looks okay

sub_fname = SUB_DIR/'basic.csv'

sub_df.to_csv(sub_fname, index=None)



!head {sub_fname}
# Use FileLink to download the file

from IPython.display import FileLink



FileLink(sub_fname)
def make_sub(fname):

    preds = model(test_ids)

    probs = sigmoid(preds)

    labels = list(map(make_labels, probs))

    sub_df = pd.DataFrame({ 'Id': test_ids, 'Predicted': labels}, columns=['Id', 'Predicted'])

    fpath = SUB_DIR/fname

    sub_df.to_csv(fpath, index=None)

    !head {fpath}

    return FileLink(fpath)
make_sub('best_submission.csv')
!pip install jovian --upgrade -q
import jovian
jovian.commit()
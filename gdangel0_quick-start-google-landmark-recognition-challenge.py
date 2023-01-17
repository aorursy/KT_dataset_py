# Load useful python libraries
import numpy as np
import pandas as pd 

# Set training and testing files
training_file = '../input/train.csv'
testing_file = '../input/test.csv'

# Load datasets
train_data = pd.read_csv(training_file)
test_data = pd.read_csv(testing_file)

print("Data successfully loaded!")
# Number of training examples
n_train = train_data.shape[0]

# Number of testing examples
n_test = test_data.shape[0]

# Proportion of training and testing data
train_per = n_train / (n_train + n_test) * 100
test_per = n_test / (n_train + n_test) * 100

# Shape of the training and testing datasets
train_shape = train_data.shape
test_shape = test_data.shape

# Unique landmarks in the dataset
n_classes = len(train_data['landmark_id'].unique())

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Proportion training/testing data = {0:.2f}% / {1:.2f}%".format(train_per, test_per))
print("Training data shape =", train_shape)
print("Testing data shape =", test_shape)
print("Number of classes =", n_classes)
train_data.head()
test_data.head()
import sys, os, requests
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

def download_image(key, url):
    # Define output directory and filename
    out_dir = os.getcwd() + '/../output/'
    filename = os.path.join(out_dir, '{}.jpg'.format(key))
    
    # Image already downloaded?
    if not os.path.exists(filename):
        # Download image data
        try:
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:  
                f.write(r.content)
        except:
            print('Could not download image {0} from {1}'.format(key, url))
            return
    
    # Open image
    try:
        image = plt.imread(filename)
    except:
        print('Failed to open image {}'.format(key))
        return
    
    return image
def explore_dataset(data, sample_id):
    """
    Visualize the Google Landmark Recognition Dataset
    """
    
    # Check sample ID
    data_len = data.shape[0]
    if not (0 <= sample_id < data_len):
        print("{} samples in dataset. {} is out of range.".format(data_len, sample_id))
        return None
    
    # Count the number of images by landmark ID
    landmark_counts = pd.DataFrame(data['landmark_id'].value_counts())
    landmark_counts.reset_index(inplace=True)
    landmark_counts.columns = ['landmark_id', 'counts']
    
    print("First 20 most represented landmarks ID:\n")
    print(landmark_counts.head(20))
    
    # Plot counts of landmarks ID in descending order
    plt.figure(figsize=(12,12))
    sns.set(font_scale=1.3)
    sns.barplot(x='landmark_id', y='counts', data=landmark_counts.head(20))
    plt.title('Most represented landmark ID')
    plt.xlabel('Landmark ID')
    plt.ylabel('# Samples')
    plt.tight_layout()
    plt.show()
    
    # Visualize dataset distribution
    plt.figure(figsize=(12,12))
    sns.distplot(data['landmark_id'])
    plt.title("Dataset distribution")
    plt.show()
    
    # Visualize one sample from dataset
    sample_data = data.iloc[sample_id]
    sample_id = sample_data['id']
    sample_url = sample_data['url']
    sample_label = sample_data['landmark_id']
    
    sample_image = download_image(sample_id, sample_url)
    
    if sample_image is not None:
        print('\nExample of Image {}:'.format(sample_id))
        print('> Min Value: {}, Max Value: {}'.format(sample_image.min(), sample_image.max()))
        print('> Shape: {}'.format(sample_image.shape))
        print('> URL: {}'.format(sample_url))
        print('> Landmark ID: {}'.format(sample_label))

        plt.figure()
        plt.axis('off')
        plt.imshow(sample_image)
# Visualize training dataset
print('--TRAIN DATASET\n')
explore_dataset(train_data, 0)
from sklearn.utils import shuffle

def display_image_examples(data, labels, n_images_per_label = 5):
    data_shuffled = shuffle(data)
    
    fig = plt.figure(figsize=(n_images_per_label, len(labels)))
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images_per_label * len(labels))
    
    for i in range(len(labels)):
        
        # Choose the first 'n_images_per_label' indexes for each label
        df = data[data['landmark_id'] == labels[i]]
        df = df[:n_images_per_label]
        
        # Display corresponding images in data
        count = 0
        for index, row in df.iterrows():
            # Download image from url
            image = download_image(row['id'], row['url'])
            # Plot it
            count += 1
            plt.subplot(len(labels), n_images_per_label, count+(i*n_images_per_label))
            plt.imshow(image)
            plt.axis('off')
import random
from time import time

# Generate 5 random landmarks ID
random.seed(time())
landmarks = []
for i in range(5):
    landmarks.append(random.randint(1, 14950))

# Display some images from the training dataset for the previous generated landmarks ID
display_image_examples(train_data, landmarks, n_images_per_label = 3)

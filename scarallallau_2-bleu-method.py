try:
  # %tensorflow_version only exists in Colab.
  import tensorflow.compat.v2 as tf
except Exception:
  pass
tf.enable_v2_behavior()

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#from __future__ import absolute_import, division, print_function, unicode_literals
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle
from tqdm.auto import tqdm
import csv
import pandas as pd
annotation_file = './annotations/captions_train2014.json'

PATH = './train2014'
# Read the json file
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

# Store captions and image names in vectors
all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = os.path.join(PATH, 'COCO_train2014_' + '%012d.jpg' % (image_id))

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

# Shuffle captions and image_names together
# Set a random state
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 30000 captions from the shuffled set
num_examples = 30000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path
image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
# Get unique images
encode_train = sorted(set(img_name_vector))

# Feel free to change batch_size according to your system configuration
image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
image_dataset = image_dataset.map(
  load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

for img, path in tqdm(image_dataset):
  batch_features = image_features_extract_model(img)
  batch_features = tf.reshape(batch_features,
                              (batch_features.shape[0], -1, batch_features.shape[3]))

  for bf, p in zip(batch_features, path):
    path_of_feature = p.numpy().decode("utf-8")
    np.save(path_of_feature, bf.numpy())
# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


# Choose the top 5000 words from the vocabulary
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(train_captions)
#train_seqs = tokenizer.texts_to_sequences(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
train_seqs = tokenizer.texts_to_sequences(train_captions)

# Pad each vector to the max_length of the captions
# If you do not provide a max_length value, pad_sequences calculates it automatically
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

# Calculates the max_length, which is used to store the attention weights
max_length = calc_max_length(train_seqs)
# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.0333,
                                                                    random_state=0)
all_captions = pd.read_csv("all_captions.csv", sep=',') 

real_captions = [x.split() for x in all_captions['true_caption'].tolist()]
pred_captions = [x.split() for x in all_captions['pred_caption'].tolist()]
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import string

import warnings
import cv2
warnings.filterwarnings("ignore")
nltk.download('punkt')
nltk.download('stopwords')
stopword_punct = list(stopwords.words('english')) + list(string.punctuation)
def get_similar_result_bleu(idx, real_captions, pred_captions):

    real_cap = real_captions[idx]
    b_score_list = []

    for idx_2 in range(len(pred_captions)):

      # eliminate stopwords and punctuation before compute bleu score
      sentence1 = [w for w in real_cap if not w in stopword_punct]
      sentence2 = [w for w in pred_captions[idx_2] if not w in stopword_punct]
      
      b_score = nltk.translate.bleu_score.sentence_bleu([sentence1], sentence2)
      b_score_list.append((idx_2, b_score))

    b_score_list.sort(key=lambda x: x[1], reverse=True)

    return b_score_list
def create_submission_file(top_k, img_name_val, real_captions, pred_captions):

    with open('./submission_bleu.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["caption", "image_list"])

        for idx in tqdm(range(len(img_name_val))):

            b_score_res = get_similar_result_bleu(idx, real_captions, pred_captions)

            writer.writerow([' '.join(real_captions[idx]), ' '.join(list(map(lambda x: str(x[0]), b_score_res[:top_k])))])
create_submission_file(len(img_name_val), img_name_val, real_captions, pred_captions)
def show_image(image_fname, new_figure=True):
  if new_figure:
    plt.figure()
  np_img = cv2.imread(image_fname)
  np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
  plt.imshow(np_img) 

def show_qualitative_results(idx1, top_k=20):

    b_score_res = get_similar_result_bleu(idx1, real_captions, pred_captions)

    print("Real capt:", ' '.join(real_captions[idx1]))
    print("Pred capt:", ' '.join(pred_captions[idx1]))
    sentence1 = [w for w in real_captions[idx1] if not w in stopword_punct]
    sentence2 = [w for w in pred_captions[idx1] if not w in stopword_punct]
    ss = nltk.translate.bleu_score.sentence_bleu([sentence1], sentence2)
    print("Score with True Predicted caption:", ss)
    print()

    show_image(img_name_val[idx1], new_figure=False)
    plt.grid(False)
    plt.ioff()
    plt.axis('off')


    fig = plt.figure(figsize=(10, 7))

    for idx2, (idx, sim_val) in enumerate(b_score_res[:20]):
        print(idx, sim_val, ' '.join(pred_captions[idx]))
        plt.subplot(4, 5, idx2+1)
        show_image(img_name_val[idx], new_figure=False)
        plt.grid(False)
        plt.ioff()
        plt.axis('off')
        plt.title('{}'.format(idx2+1))
show_qualitative_results(idx1 = 0)
all_idx = []
top_k = 1000

for ref_idx in tqdm(range(len(img_name_val))):
    b_score_res = get_similar_result_bleu(ref_idx, real_captions, pred_captions)
    list_res = list(map(lambda x: x[0], b_score_res[:top_k]))
    index = list_res.index(ref_idx)
    all_idx.append(index)

n, bins, patches = plt.hist(all_idx, bins=1000)
plt.xlabel('top K')
plt.ylabel('Frequency')

plt.show()

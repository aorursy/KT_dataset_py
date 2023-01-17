!pip install "tensorflow>=2"

!pip install "tensorflow_hub>=0.7"

!pip install bert-for-tf2

!pip install sentencepiece
import tensorflow as tf

import tensorflow_hub as hub

print("TF version: ", tf.__version__)

print("Hub version: ", hub.__version__)
import tensorflow_hub as hub

import tensorflow as tf

import bert

FullTokenizer = bert.bert_tokenization.FullTokenizer

from tensorflow.keras.models import Model       # Keras is the new high level API for TensorFlow

import math

import numpy as np
max_seq_length = 512  # Your choice here.

input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                       name="input_word_ids")

input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                   name="input_mask")

segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,

                                    name="segment_ids")

bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",

                            trainable=True)

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])



model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])



vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = FullTokenizer(vocab_file, do_lower_case)
# See BERT paper: https://arxiv.org/pdf/1810.04805.pdf

# And BERT implementation convert_single_example() at https://github.com/google-research/bert/blob/master/run_classifier.py



def get_masks(tokens, max_seq_length):

    """Mask for padding"""

    if len(tokens)>max_seq_length:

        raise IndexError("Token length more than max seq length!")

    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))





def get_segments(tokens, max_seq_length):

    """Segments: 0 for the first sequence, 1 for the second"""

    if len(tokens)>max_seq_length:

        raise IndexError("Token length more than max seq length!")

    segments = []

    current_segment_id = 0

    for token in tokens:

        segments.append(current_segment_id)

        if token == "[SEP]":

            current_segment_id = 1

    return segments + [0] * (max_seq_length - len(tokens))





def get_ids(tokens, tokenizer, max_seq_length):

    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))

    return input_ids
def tokenize_sentence(sentence):

    stokens = tokenizer.tokenize(sentence)

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    

    input_ids = get_ids(stokens, tokenizer, max_seq_length)

    input_masks = get_masks(stokens, max_seq_length)

    input_segments = get_segments(stokens, max_seq_length)

    

    return input_ids, input_masks, input_segments



def compare_sentences(sentence_1, sentence_2, distance_metric):

    input_ids_1, input_masks_1, input_segments_1 = tokenize_sentence(sentence_1)

    input_ids_2, input_masks_2, input_segments_2 = tokenize_sentence(sentence_2)

    

    pool_embs_1, all_embs_1 = model.predict([[input_ids_1],[input_masks_1],[input_segments_1]])

    pool_embs_2, all_embs_2 = model.predict([[input_ids_2],[input_masks_2],[input_segments_2]])

#     print(pool_embs_1, all_embs_1)

#     print(pool_embs_2, all_embs_2)

    return distance_metric(pool_embs_1[0], pool_embs_2[0])

    

def square_rooted(x):

    return math.sqrt(sum([a*a for a in x]))



def cosine_similarity(x,y):

    numerator = sum(a*b for a,b in zip(x,y))

    denominator = square_rooted(x)*square_rooted(y)

    return numerator/float(denominator)



def dummy_metric(x,y):

    return 42
s = []

s.append('How are you doing?')

s.append('How much are we feeling?')

s.append('What are you doing?')

s.append('What`s up?')

s.append('Are you doing?')

central = s[0]

print("Central phrase: '{}'".format(central))

for sentence in s:

    print("Distance to '{}' = {}".format(sentence, round(compare_sentences(central, sentence, cosine_similarity), 3)))
w = []

w.append('data')

w.append('Data')

w.append('Data Set')

w.append('datadata')

w.append('dataset')

w.append('DataFrame')

w.append('dataframe')

w.append('df')

w.append('pd.DataFrame')

central = w[0]

print("Central phrase: '{}'".format(central))

for word in w:

    print("Distance to '{}' = {}".format(word, round(compare_sentences(central, word, cosine_similarity), 3)))
import numpy as np
c = []

cnot = []

c.append("""

def apply_window(image, center, width):

    image = image.copy()



    min_value = center - width // 2

    max_value = center + width // 2



    image[image < min_value] = min_value

    image[image > max_value] = max_value



    return image

""")

c.append("""

def image_windowed(image, custom_center=50, custom_width=130, out_side_val=False):

    '''

    Important thing to note in this function: The image migth be changed in place!

    '''

    # see: https://www.kaggle.com/allunia/rsna-ih-detection-eda-baseline

    min_value = custom_center - (custom_width/2)

    max_value = custom_center + (custom_width/2)

    

    # Including another value for values way outside the range, to (hopefully) make segmentation processes easier. 

    out_value_min = custom_center - custom_width

    out_value_max = custom_center + custom_width

    

    if out_side_val:

        image[np.logical_and(image < min_value, image > out_value_min)] = min_value

        image[np.logical_and(image > max_value, image < out_value_max)] = max_value

        image[image < out_value_min] = out_value_min

        image[image > out_value_max] = out_value_max

    

    else:

        image[image < min_value] = min_value

        image[image > max_value] = max_value

    

    return image

""")

c.append("""

def image_crop(image):

    # Based on this stack overflow post: https://stackoverflow.com/questions/26310873/how-do-i-crop-an-image-on-a-white-background-with-python

    mask = image == 0



    # Find the bounding box of those pixels

    coords = np.array(np.nonzero(~mask))

    top_left = np.min(coords, axis=1)

    bottom_right = np.max(coords, axis=1)



    out = image[top_left[0]:bottom_right[0],

                top_left[1]:bottom_right[1]]

    

    return out

""")

cnot.append("""

def normalize_minmax(img):

    mi, ma = img.min(), img.max()

    return (img - mi) / (ma - mi)""")

cnot.append("""def normalize(img, means, stds, tensor=False):

    return (img - means)/stds""")

cnot.append("""X_train = X_train / 255.0

test = test / 255.0

mean_px = X_train.mean().astype(np.float32)

std_px = X_train.std().astype(np.float32)""")

cnot.append("""def standardize(x): 

    return (x-mean_px)/std_px""")

cnot.append("""def rle_decode(mask_rle, shape=(768, 768)):

    s = mask_rle.split()

    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

    starts -= 1

    ends = starts + lengths

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)

    for lo, hi in zip(starts, ends):

        img[lo:hi] = 1

    return img.reshape(shape).T  # Needed to align to RLE direction""")

cnot.append("""def mask2rle(img):

    pixels = img.T.flatten()

    pixels = np.concatenate([[0], pixels, [0]])

    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1

    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)""")

cnot.append("""def deskew(img):

    m = cv2.moments(img)

    if abs(m['mu02']) < 1e-2:

        return img.copy()

    skew = m['mu11']/m['mu02']

    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])

    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)

    return img""")

cnot.append("""image = imageio.imread('../input/image1.jpg')

image_rotated = rotate.augment_images([image])

image_noise = gaussian_noise.augment_images([image])

image_crop = crop.augment_images([image])

image_hue = hue.augment_images([image])

image_trans = elastic_trans.augment_images([image])

image_coarse = coarse_drop.augment_images([image])""")



central = c[0]

results_c = []

for chunk in c:

    results_c.append(compare_sentences(central, chunk, cosine_similarity))



results_cnot = []

for chunk in cnot:

    results_cnot.append(compare_sentences(central, chunk, cosine_similarity))

print(np.mean(results_c), np.mean(results_cnot))
import pandas as pd
nl2ml = pd.read_csv('../input/nl2ml-images/nl2ml_images.csv')
cat_names = nl2ml['Preprocessing class (for doc about methods)'].value_counts().head(5).keys().tolist()
central = nl2ml[(nl2ml['Preprocessing class (for doc about methods)'] == cat_names[0])]['Code'].reset_index(drop=True)[0]

results = {"central":cat_names[0]}



for cat in cat_names:

    rows = nl2ml[(nl2ml['Preprocessing class (for doc about methods)'] == cat)].reset_index(drop=True)

    chunks = rows['Code']

    results_c = []

    for chunk in chunks:

        if len(chunk) <= 512:

            results_c.append(compare_sentences(central, chunk, cosine_similarity))

        else: continue

    results.update({cat:round(np.mean(results_c), 3)})

    del results_c
results